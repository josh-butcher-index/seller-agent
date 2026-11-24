"""Bearer token providers for REST clients, including tokens that expire.

Subclass RefreshingTokenProvider and implement `_login()` and `_refresh()` for an
auth API; BearerTokenAuth sends the token and retries once on a 401.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable, Generator, Hashable
from dataclasses import dataclass
from typing import Optional, TypeVar

import httpx

from .ssp_base import SSPAuthenticationError

logger = logging.getLogger(__name__)


class CredentialsRejectedError(SSPAuthenticationError):
    """The auth API rejected the configured credentials (or refresh token) outright."""


@dataclass(frozen=True)
class TokenGrant:
    """Tokens returned by a login or refresh call."""

    access_token: str
    expires_in: int
    refresh_token: Optional[str] = None


class TokenProvider(ABC):
    """Supplies the bearer token for outgoing requests."""

    @abstractmethod
    async def get_token(self, *, rejected_token: Optional[str] = None) -> str:
        """Return a usable token, replacing `rejected_token` if the server returned 401 for it."""


class StaticTokenProvider(TokenProvider):
    """A fixed token, e.g. a long-lived API key."""

    def __init__(self, token: str) -> None:
        self._token = token

    async def get_token(self, *, rejected_token: Optional[str] = None) -> str:
        return self._token


class RefreshingTokenProvider(TokenProvider):
    """Caches an expiring token, renewing it by refresh or login; safe across threads and loops.

    A rejected login is never retried, so a wrong password can't trip brute-force detection.
    """

    def __init__(
        self,
        *,
        provider_name: str,
        refresh_margin_seconds: int = 60,
        initial_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.provider_name = provider_name
        self._clock = clock
        self._refresh_margin_seconds = refresh_margin_seconds
        self._initial_backoff_seconds = initial_backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds

        self._state_lock = threading.Lock()
        self._loop_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
            weakref.WeakKeyDictionary()
        )
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._refresh_at = 0.0
        self._expires_at = 0.0
        self._credentials_rejected_message: Optional[str] = None
        self._backoff_seconds = 0.0
        self._retry_login_at = 0.0
        self._last_login_error: Optional[SSPAuthenticationError] = None

    @abstractmethod
    async def _login(self) -> TokenGrant:
        """Log in; raise CredentialsRejectedError if rejected, else SSPAuthenticationError."""

    @abstractmethod
    async def _refresh(self, refresh_token: str) -> TokenGrant:
        """Exchange a refresh token for new tokens; raise SSPAuthenticationError on failure."""

    async def get_token(self, *, rejected_token: Optional[str] = None) -> str:
        token = self._cached_token(rejected_token)
        if token is not None:
            return token

        async with self._loop_lock():
            # Another caller on this loop may have renewed the token while we waited.
            token = self._cached_token(rejected_token)
            if token is not None:
                return token
            try:
                grant = await self._renew()
            except CredentialsRejectedError:
                raise
            except SSPAuthenticationError as exc:
                token = self._unexpired_token(rejected_token)
                if token is None:
                    raise
                logger.warning(
                    "%s token renewal failed (%s); using the current token until it expires",
                    self.provider_name,
                    exc,
                )
                return token
            self._store(grant)
            return grant.access_token

    def _cached_token(self, rejected_token: Optional[str]) -> Optional[str]:
        with self._state_lock:
            self._raise_if_credentials_rejected()
            if self._access_token is None or self._access_token == rejected_token:
                return None
            if self._clock() >= self._refresh_at:
                return None
            return self._access_token

    def _unexpired_token(self, rejected_token: Optional[str]) -> Optional[str]:
        with self._state_lock:
            if self._access_token is None or self._access_token == rejected_token:
                return None
            if self._clock() >= self._expires_at:
                return None
            return self._access_token

    def _raise_if_credentials_rejected(self) -> None:
        # A new instance per raise: re-raising a stored one grows its __traceback__ every time.
        if self._credentials_rejected_message is not None:
            raise CredentialsRejectedError(self._credentials_rejected_message)

    def _loop_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        with self._state_lock:
            lock = self._loop_locks.get(loop)
            if lock is None:
                lock = asyncio.Lock()
                self._loop_locks[loop] = lock
            return lock

    async def _renew(self) -> TokenGrant:
        with self._state_lock:
            refresh_token = self._refresh_token

        if refresh_token:
            try:
                return await self._refresh(refresh_token)
            except CredentialsRejectedError as exc:
                logger.info(
                    "%s refresh token was rejected (%s); logging in again", self.provider_name, exc
                )
                with self._state_lock:
                    # Another event loop may have stored a newer refresh token meanwhile.
                    if self._refresh_token == refresh_token:
                        self._refresh_token = None
            except SSPAuthenticationError as exc:
                logger.info("%s token refresh failed (%s); trying a login", self.provider_name, exc)

        return await self._login_with_backoff()

    async def _login_with_backoff(self) -> TokenGrant:
        with self._state_lock:
            self._raise_if_credentials_rejected()
            wait_seconds = self._retry_login_at - self._clock()
            if wait_seconds > 0 and self._last_login_error is not None:
                raise SSPAuthenticationError(
                    f"{self._last_login_error}; next {self.provider_name} login attempt "
                    f"in {wait_seconds:.0f}s"
                )

        try:
            return await self._login()
        except CredentialsRejectedError as exc:
            with self._state_lock:
                self._credentials_rejected_message = str(exc)
            logger.error(
                "%s login was rejected; no further login attempts until the "
                "credentials are fixed and the service is restarted",
                self.provider_name,
            )
            raise
        except SSPAuthenticationError as exc:
            with self._state_lock:
                self._backoff_seconds = min(
                    max(self._backoff_seconds * 2, self._initial_backoff_seconds),
                    self._max_backoff_seconds,
                )
                self._retry_login_at = self._clock() + self._backoff_seconds
                self._last_login_error = exc
            raise

    def _store(self, grant: TokenGrant) -> None:
        margin = min(self._refresh_margin_seconds, grant.expires_in // 2)
        with self._state_lock:
            self._access_token = grant.access_token
            self._refresh_token = grant.refresh_token or self._refresh_token
            self._refresh_at = self._clock() + grant.expires_in - margin
            self._expires_at = self._clock() + grant.expires_in
            self._backoff_seconds = 0.0
            self._retry_login_at = 0.0
            self._last_login_error = None


_ProviderT = TypeVar("_ProviderT", bound=TokenProvider)

_shared_providers: dict[Hashable, TokenProvider] = {}
_shared_providers_lock = threading.Lock()


def shared_token_provider(key: Hashable, create: Callable[[], _ProviderT]) -> _ProviderT:
    """Return the process-wide provider for `key`, creating it on first use.

    SSP clients are built per operation, so sharing keeps one token cache per credential.
    """
    with _shared_providers_lock:
        provider = _shared_providers.get(key)
        if provider is None:
            provider = create()
            _shared_providers[key] = provider
        return provider  # type: ignore[return-value]


def clear_shared_token_providers() -> None:
    """Forget all shared providers (for tests)."""
    with _shared_providers_lock:
        _shared_providers.clear()


class BearerTokenAuth(httpx.Auth):
    """httpx auth that sends a provider's token and retries once on a 401."""

    requires_request_body = True

    def __init__(
        self,
        token_provider: TokenProvider,
        *,
        header: str = "Authorization",
        prefix: str = "Bearer",
    ) -> None:
        self._token_provider = token_provider
        self._header = header
        self._prefix = prefix

    def sync_auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        raise RuntimeError("BearerTokenAuth supports httpx.AsyncClient only")

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        token = await self._token_provider.get_token()
        request.headers[self._header] = f"{self._prefix} {token}"
        response = yield request
        if response.status_code != 401:
            return

        new_token = await self._token_provider.get_token(rejected_token=token)
        if new_token == token:
            return
        request.headers[self._header] = f"{self._prefix} {new_token}"
        response = yield request
        if response.status_code == 401:
            raise SSPAuthenticationError(
                f"{request.url.host} returned 401 again after the token was renewed, for "
                f"{request.method} {request.url.path}"
            )
