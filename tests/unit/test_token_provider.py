"""Unit tests for bearer token providers and Index Exchange token refresh."""

import asyncio
import json
import logging
import traceback

import httpx
import pytest
from pydantic import SecretStr

from ad_seller.clients.ssp_base import SSPAuthenticationError
from ad_seller.clients.ssp_index_exchange import (
    IndexExchangeSSPClient,
    IndexExchangeTokenProvider,
)
from ad_seller.clients.token_provider import (
    BearerTokenAuth,
    CredentialsRejectedError,
    RefreshingTokenProvider,
    StaticTokenProvider,
    TokenGrant,
    clear_shared_token_providers,
)

BASE_URL = "https://ix.test"
PASSWORD = "s3cret-pa55word"
DEALS_PATH = "/api/deals/v3/deals"


@pytest.fixture(autouse=True)
def _clear_shared_providers():
    clear_shared_token_providers()
    yield
    clear_shared_token_providers()


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeIndex:
    """Routes auth and deals requests; tokens are numbered in issue order."""

    def __init__(self, *, expires_in: int = 3600) -> None:
        self.expires_in = expires_in
        self.issued = 0
        self.valid_tokens: set[str] = set()
        self.valid_refresh_tokens: set[str] = set()
        self.calls: list[str] = []
        self.login_status = 200
        self.refresh_status: int | None = None
        self.deals_status: int | None = None

    def _issue(self) -> dict:
        self.issued += 1
        access, refresh = f"access-{self.issued}", f"refresh-{self.issued}"
        self.valid_tokens = {access}
        self.valid_refresh_tokens = {refresh}
        return {
            "access_token": access,
            "expires_in": self.expires_in,
            "refresh_token": refresh,
            "token_type": "Bearer",
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(path)
        if path.endswith("/login"):
            if self.login_status != 200:
                return httpx.Response(self.login_status, json={"message": "Unauthorized"})
            return httpx.Response(
                200, json={"loginResponse": {"redir": "deals", "authResponse": self._issue()}}
            )
        if path.endswith("/refresh"):
            body = json.loads(request.content)
            if self.refresh_status is not None:
                return httpx.Response(self.refresh_status, json={"message": "nope"})
            if body.get("refreshToken") not in self.valid_refresh_tokens:
                return httpx.Response(401, json={"message": "Unauthorized"})
            return httpx.Response(200, json=self._issue())
        if self.deals_status is not None:
            return httpx.Response(self.deals_status)
        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if token not in self.valid_tokens:
            return httpx.Response(401)
        return httpx.Response(200, json={"deals": [], "token": token})

    def count(self, suffix: str) -> int:
        return sum(1 for c in self.calls if c.endswith(suffix))


def _provider(index: FakeIndex, clock: FakeClock | None = None) -> IndexExchangeTokenProvider:
    provider = IndexExchangeTokenProvider(
        base_url=BASE_URL,
        username="agent@example.com",
        password=SecretStr(PASSWORD),
        transport=httpx.MockTransport(index.handler),
    )
    if clock is not None:
        provider._clock = clock
    return provider


def _deals_client(index: FakeIndex, provider) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=BASE_URL,
        auth=BearerTokenAuth(provider),
        transport=httpx.MockTransport(index.handler),
    )


class TestIndexExchangeTokenLifecycle:
    @pytest.mark.asyncio
    async def test_first_request_logs_in_and_later_requests_reuse_the_token(self):
        index = FakeIndex()
        provider = _provider(index)
        async with _deals_client(index, provider) as http:
            first = await http.get(DEALS_PATH)
            second = await http.get(DEALS_PATH)

        assert first.status_code == second.status_code == 200
        assert second.json()["token"] == "access-1"
        assert index.count("/login") == 1

    @pytest.mark.asyncio
    async def test_login_sends_username_and_password(self):
        sent = []

        def handler(request):
            sent.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={"loginResponse": {"authResponse": {"access_token": "a", "expires_in": 60}}},
            )

        provider = IndexExchangeTokenProvider(
            base_url=BASE_URL,
            username="agent@example.com",
            password=SecretStr(PASSWORD),
            transport=httpx.MockTransport(handler),
        )
        assert await provider.get_token() == "a"
        assert sent == [{"username": "agent@example.com", "password": PASSWORD}]

    @pytest.mark.asyncio
    async def test_token_near_expiry_is_refreshed_and_rotated_refresh_token_is_used(self):
        index = FakeIndex(expires_in=300)
        clock = FakeClock()
        provider = _provider(index, clock)

        assert await provider.get_token() == "access-1"
        clock.now += 300 - 60  # inside the refresh margin
        assert await provider.get_token() == "access-2"
        clock.now += 300 - 60
        assert await provider.get_token() == "access-3"

        assert index.count("/login") == 1
        assert index.count("/refresh") == 2

    @pytest.mark.asyncio
    async def test_rejected_refresh_token_falls_back_to_login(self):
        index = FakeIndex(expires_in=300)
        clock = FakeClock()
        provider = _provider(index, clock)

        await provider.get_token()
        index.valid_refresh_tokens = set()  # refresh token expired server-side
        clock.now += 300

        assert await provider.get_token() == "access-2"
        assert index.calls[-2:] == [
            "/api/authentication/v1/refresh",
            "/api/authentication/v1/login",
        ]

    @pytest.mark.asyncio
    async def test_failed_refresh_falls_back_to_login(self):
        index = FakeIndex(expires_in=300)
        clock = FakeClock()
        provider = _provider(index, clock)

        await provider.get_token()
        index.refresh_status = 500
        clock.now += 300

        assert await provider.get_token() == "access-2"
        assert index.count("/login") == 2

    @pytest.mark.asyncio
    async def test_401_from_api_renews_token_and_retries_once(self):
        index = FakeIndex()
        provider = _provider(index)
        async with _deals_client(index, provider) as http:
            await http.get(DEALS_PATH)
            index.valid_tokens = set()  # token revoked server-side
            index.valid_refresh_tokens = {"refresh-1"}
            response = await http.post(DEALS_PATH, json={"name": "deal"})

        assert response.status_code == 200
        assert response.json()["token"] == "access-2"
        assert index.count(DEALS_PATH) == 3

    @pytest.mark.asyncio
    async def test_401_after_renewing_the_token_raises_authentication_error(self):
        index = FakeIndex()
        provider = _provider(index)
        async with _deals_client(index, provider) as http:
            await http.get(DEALS_PATH)
            index.deals_status = 401
            with pytest.raises(
                SSPAuthenticationError, match="401 again after the token was renewed"
            ):
                await http.get(DEALS_PATH)

        assert index.count(DEALS_PATH) == 3

    @pytest.mark.asyncio
    async def test_concurrent_requests_share_one_refresh(self):
        index = FakeIndex(expires_in=300)
        clock = FakeClock()
        provider = _provider(index, clock)
        await provider.get_token()
        clock.now += 300

        tokens = await asyncio.gather(*(provider.get_token() for _ in range(10)))

        assert set(tokens) == {"access-2"}
        assert index.count("/refresh") == 1

    @pytest.mark.asyncio
    async def test_concurrent_401s_with_the_same_token_share_one_renewal(self):
        index = FakeIndex()
        provider = _provider(index)
        await provider.get_token()

        tokens = await asyncio.gather(
            *(provider.get_token(rejected_token="access-1") for _ in range(10))
        )

        assert set(tokens) == {"access-2"}
        assert index.count("/refresh") == 1


class TestRenewalFailureHandling:
    @pytest.mark.asyncio
    async def test_unexpired_token_is_used_when_renewal_fails(self, caplog):
        index = FakeIndex(expires_in=300)
        clock = FakeClock()
        provider = _provider(index, clock)
        await provider.get_token()
        index.refresh_status = 503
        index.login_status = 503
        clock.now += 250  # inside the refresh margin, before expiry

        with caplog.at_level(logging.WARNING):
            assert await provider.get_token() == "access-1"
        assert "using the current token until it expires" in caplog.text

        clock.now += 50  # now actually expired
        with pytest.raises(SSPAuthenticationError):
            await provider.get_token()

    @pytest.mark.asyncio
    async def test_rejected_token_is_not_reused_when_renewal_fails(self):
        index = FakeIndex(expires_in=300)
        provider = _provider(index)
        await provider.get_token()
        index.refresh_status = 503
        index.login_status = 503

        with pytest.raises(SSPAuthenticationError):
            await provider.get_token(rejected_token="access-1")

    @pytest.mark.asyncio
    async def test_transient_refresh_failure_keeps_the_refresh_token(self):
        index = FakeIndex(expires_in=300)
        clock = FakeClock()
        provider = _provider(index, clock)
        await provider.get_token()
        index.refresh_status = 503
        index.login_status = 503
        clock.now += 250
        await provider.get_token()  # falls back to the unexpired token

        index.refresh_status = None
        index.login_status = 200
        clock.now += 10
        assert await provider.get_token() == "access-2"
        assert index.calls[-1] == "/api/authentication/v1/refresh"
        assert index.count("/login") == 2  # the initial login and the one failed attempt

    @pytest.mark.asyncio
    async def test_rejected_refresh_does_not_clear_a_newer_refresh_token(self):
        class RacingProvider(RefreshingTokenProvider):
            async def _login(self):
                return TokenGrant(access_token="from-login", expires_in=300)

            async def _refresh(self, refresh_token):
                # Another event loop stores a newer refresh token while this refresh fails.
                self._refresh_token = "RT2"
                raise CredentialsRejectedError("refresh token rejected")

        provider = RacingProvider(provider_name="Test")
        provider._refresh_token = "RT1"

        await provider.get_token()
        assert provider._refresh_token == "RT2"

    @pytest.mark.asyncio
    async def test_rejected_login_raises_a_new_exception_each_time(self):
        index = FakeIndex()
        index.login_status = 401
        provider = _provider(index)

        raised = []
        for _ in range(20):
            with pytest.raises(CredentialsRejectedError) as exc_info:
                await provider.get_token()
            raised.append(exc_info.value)

        assert len({id(exc) for exc in raised}) == 20
        assert {str(exc) for exc in raised} == {str(raised[0])}
        depth = [len(traceback.extract_tb(exc.__traceback__)) for exc in raised]
        assert depth[-1] == depth[1]


def test_shared_provider_works_across_event_loops():
    index = FakeIndex(expires_in=300)
    clock = FakeClock()
    provider = _provider(index, clock)

    assert asyncio.run(provider.get_token()) == "access-1"
    clock.now += 300
    assert asyncio.run(provider.get_token()) == "access-2"
    assert index.count("/refresh") == 1


class TestIndexExchangeLoginFailures:
    @pytest.mark.asyncio
    async def test_rejected_login_is_a_clear_error_without_secrets(self, caplog):
        index = FakeIndex()
        index.login_status = 401
        provider = _provider(index)

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(CredentialsRejectedError) as exc_info:
                await provider.get_token()

        message = str(exc_info.value)
        assert "login was rejected (401)" in message
        assert "INDEX_EXCHANGE_PASSWORD" in message
        assert PASSWORD not in message
        assert PASSWORD not in caplog.text

    @pytest.mark.asyncio
    async def test_rejected_login_is_never_retried(self):
        index = FakeIndex()
        index.login_status = 401
        provider = _provider(index)

        for _ in range(5):
            with pytest.raises(CredentialsRejectedError):
                await provider.get_token()
        index.login_status = 200  # even if the server would now accept it
        with pytest.raises(CredentialsRejectedError):
            await provider.get_token()

        assert index.count("/login") == 1

    @pytest.mark.asyncio
    async def test_transient_login_failure_backs_off(self):
        index = FakeIndex()
        index.login_status = 503
        clock = FakeClock()
        provider = _provider(index, clock)

        with pytest.raises(SSPAuthenticationError, match="HTTP 503"):
            await provider.get_token()
        with pytest.raises(SSPAuthenticationError, match="next Index Exchange login attempt"):
            await provider.get_token()
        assert index.count("/login") == 1

        index.login_status = 200
        clock.now += 1
        assert await provider.get_token() == "access-1"
        assert index.count("/login") == 2

    @pytest.mark.asyncio
    async def test_network_error_is_an_authentication_error(self):
        def handler(request):
            raise httpx.ConnectError("boom", request=request)

        provider = IndexExchangeTokenProvider(
            base_url=BASE_URL,
            username="agent@example.com",
            password=SecretStr(PASSWORD),
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(SSPAuthenticationError, match="login request failed: ConnectError"):
            await provider.get_token()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body",
        [
            {"loginResponse": {"authResponse": {"expires_in": 60}}},
            {"loginResponse": {"authResponse": {"access_token": "a"}}},
            {"unexpected": True},
            ["not", "a", "dict"],
        ],
    )
    async def test_malformed_login_response_is_an_authentication_error(self, body):
        provider = IndexExchangeTokenProvider(
            base_url=BASE_URL,
            username="agent@example.com",
            password=SecretStr(PASSWORD),
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)),
        )
        with pytest.raises(SSPAuthenticationError):
            await provider.get_token()


class TestIndexExchangeTokenProviderConfig:
    @pytest.mark.parametrize(
        "base_url", ["http://app.indexexchange.com", "ftp://app.indexexchange.com"]
    )
    def test_refuses_to_send_credentials_without_https(self, base_url):
        with pytest.raises(ValueError, match="must use https"):
            IndexExchangeTokenProvider(
                base_url=base_url, username="u@example.com", password=SecretStr("p")
            )

    def test_allows_plain_http_to_localhost(self):
        IndexExchangeTokenProvider(
            base_url="http://localhost:8080", username="u@example.com", password=SecretStr("p")
        )

    def test_shared_provider_is_reused_per_credential(self):
        first = IndexExchangeSSPClient(
            base_url=BASE_URL, username="a@example.com", password=SecretStr("p")
        )
        second = IndexExchangeSSPClient(
            base_url=BASE_URL + "/", username="a@example.com", password=SecretStr("p")
        )
        other_user = IndexExchangeSSPClient(
            base_url=BASE_URL, username="b@example.com", password=SecretStr("p")
        )
        new_password = IndexExchangeSSPClient(
            base_url=BASE_URL, username="a@example.com", password=SecretStr("q")
        )

        assert first._token_provider is second._token_provider
        assert first._token_provider is not other_user._token_provider
        assert first._token_provider is not new_password._token_provider

    def test_provider_repr_does_not_expose_password(self):
        provider = IndexExchangeTokenProvider(
            base_url=BASE_URL, username="u@example.com", password=SecretStr(PASSWORD)
        )
        assert PASSWORD not in repr(provider)
        assert PASSWORD not in repr(vars(provider))

    @pytest.mark.asyncio
    async def test_connect_authenticates_requests_with_the_provider(self):
        client = IndexExchangeSSPClient(
            base_url=BASE_URL, username="a@example.com", password=SecretStr("p")
        )
        await client.connect()
        try:
            assert isinstance(client._http.auth, BearerTokenAuth)
        finally:
            await client.disconnect()


class TestStaticToken:
    @pytest.mark.asyncio
    async def test_static_token_401_is_returned_without_retry(self):
        calls = []

        def handler(request):
            calls.append(request.headers["Authorization"])
            return httpx.Response(401)

        async with httpx.AsyncClient(
            base_url=BASE_URL,
            auth=BearerTokenAuth(StaticTokenProvider("fixed")),
            transport=httpx.MockTransport(handler),
        ) as http:
            response = await http.get("/deals")

        assert response.status_code == 401
        assert calls == ["Bearer fixed"]


class TestSSPFactoryCredentials:
    def _settings(self, **overrides):
        import types

        values = {
            "ssp_connectors": "index_exchange",
            "ssp_routing_rules": "",
            "index_exchange_api_url": "https://app.indexexchange.com",
            "index_exchange_username": "",
            "index_exchange_password": None,
            "index_exchange_api_key": None,
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def test_credentials_build_a_refreshing_client(self):
        from ad_seller.clients.ssp_factory import build_ssp_registry

        registry = build_ssp_registry(
            self._settings(
                index_exchange_username="agent@example.com",
                index_exchange_password=SecretStr("pw"),
            )
        )
        client = registry.get_client("index_exchange")
        assert isinstance(client._token_provider, IndexExchangeTokenProvider)

    def test_registry_rebuilds_share_one_provider(self):
        from ad_seller.clients.ssp_factory import build_ssp_registry

        settings = self._settings(
            index_exchange_username="agent@example.com",
            index_exchange_password=SecretStr("pw"),
        )
        first = build_ssp_registry(settings).get_client("index_exchange")
        second = build_ssp_registry(settings).get_client("index_exchange")
        assert first._token_provider is second._token_provider

    def test_leftover_api_key_warns_and_skips_index(self, caplog):
        from ad_seller.clients.ssp_factory import build_ssp_registry

        with caplog.at_level(logging.WARNING):
            registry = build_ssp_registry(
                self._settings(index_exchange_api_key=SecretStr("old-token"))
            )

        assert registry.list_ssps() == []
        assert "INDEX_EXCHANGE_API_KEY is no longer used" in caplog.text
        assert "old-token" not in caplog.text

    def test_missing_credentials_skip_index(self, caplog):
        from ad_seller.clients.ssp_factory import build_ssp_registry

        with caplog.at_level(logging.WARNING):
            registry = build_ssp_registry(self._settings())

        assert registry.list_ssps() == []
        assert "INDEX_EXCHANGE_USERNAME" in caplog.text
        assert "INDEX_EXCHANGE_API_KEY" not in caplog.text
