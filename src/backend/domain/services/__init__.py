import asyncio
import ssl
import json
import xml.etree.ElementTree as ET # noqa
from enum import Enum
from typing import Any

import aiohttp

from logger import logger


class HttpRequestException(Exception):
    ...


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"


class HTTPMixin:
    def __init__(
            self,
            *args,
            max_retries: int = 3,
            retry_delay: int = 1,
            timeout: int = 30,
            ssl_verify: bool = False,
            ssl_cert_path: str | None = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.session: aiohttp.ClientSession | None = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.ssl_verify = ssl_verify
        self.ssl_cert_path = ssl_cert_path

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            connector = None
            if not self.ssl_verify:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                connector = aiohttp.TCPConnector(ssl=ctx)
            elif self.ssl_cert_path:
                connector = aiohttp.TCPConnector(ssl=self.ssl_cert_path)  # noqa

            self.session = aiohttp.ClientSession(
                headers={"Content-Type": "application/json"}, connector=connector
            )
        if self.session:
            return self.session
        else:
            raise

    async def request(
            self,
            method: HttpMethod,
            url: str,
            data: dict | None = None,
            url_params: dict | None = None,
    ) -> Any:
        session = await self._get_session()

        if method == HttpMethod.GET:
            kwargs: dict = {"params": {**(data or {}), **(url_params or {})} or None}
        elif method == HttpMethod.POST:
            kwargs = {
                "data": json.dumps(data) if data is not None else None,
                "params": url_params,
            }
        else:
            raise HttpRequestException(f"Unsupported method: {method}")

        return await self._request_with_retry(
            session,
            method.value, # noqa
            url,
            kwargs
        )

    async def _request_with_retry(
            self,
            session: aiohttp.ClientSession,
            method: str,
            url: str,
            kwargs: dict,
    ) -> Any:
        for attempt in range(self.max_retries):
            try:
                async with session.request(
                        method, url, timeout=self.timeout, **kwargs
                ) as response:
                    return await self._handle_response(response)
            except aiohttp.ClientError as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise HttpRequestException(f"Full attempts: {e}") from e
        return None

    @staticmethod
    async def _handle_response(response: aiohttp.ClientResponse) -> Any:
        text = await response.text()
        try:
            return await response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Wrong JSON response: {e}", exc_info=True)
            logger.error(f"Response: {text[:500]}")
            raise Exception(f"Wrong JSON response: {e}") from e

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()


class SOAPMixin(HTTPMixin):
    SOAP11_NS = "http://schemas.xmlsoap.org/soap/envelope/"
    SOAP12_NS = "http://www.w3.org/2003/05/soap-envelope"

    def __init__(
            self,
            *args,
            soap_version: str = "1.1",
            soap_action: str | None = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if soap_version not in ("1.1", "1.2"):
            raise ValueError("soap_version must be '1.1' or '1.2'")
        self.soap_version = soap_version
        self.soap_action = soap_action

    @property
    def _envelope_ns(self) -> str:
        return self.SOAP11_NS if self.soap_version == "1.1" else self.SOAP12_NS

    def _build_envelope(self, body: str | ET.Element) -> str:
        body_xml = (
            ET.tostring(body, encoding="unicode") if isinstance(body, ET.Element) else body
        )
        return (
            '<?xml version="1.0" encoding="utf-8"?>'
            f'<soap:Envelope xmlns:soap="{self._envelope_ns}">'
            f"<soap:Body>{body_xml}</soap:Body>"
            "</soap:Envelope>"
        )

    def _build_headers(self, action: str | None) -> dict:
        action = action or self.soap_action
        if self.soap_version == "1.1":
            return {
                "Content-Type": "text/xml; charset=utf-8",
                "SOAPAction": f'"{action}"' if action else '""',
            }
        ct = "application/soap+xml; charset=utf-8"
        if action:
            ct += f'; action="{action}"'
        return {"Content-Type": ct}

    async def soap_request(
            self,
            url: str,
            body: str | ET.Element,
            soap_action: str | None = None,
            url_params: dict | None = None,
    ) -> Any:
        session = await self._get_session()
        kwargs = {
            "data": self._build_envelope(body),
            "params": url_params,
            "headers": self._build_headers(soap_action),
        }
        return await self._request_with_retry(session, "POST", url, kwargs)

    @staticmethod
    async def _handle_response(response: aiohttp.ClientResponse) -> Any:
        text = await response.text()
        try:
            root = ET.fromstring(text)
        except ET.ParseError as e:
            logger.error(f"Wrong XML response: {e}", exc_info=True)
            logger.error(f"Response: {text[:500]}")
            raise Exception(f"Wrong XML response: {e}") from e

        for ns in (SOAPMixin.SOAP11_NS, SOAPMixin.SOAP12_NS):
            body = root.find(f"{{{ns}}}Body")
            if body is not None:
                break
        if body is None:
            return root

        for ns in (SOAPMixin.SOAP11_NS, SOAPMixin.SOAP12_NS):
            fault = body.find(f"{{{ns}}}Fault")
            if fault is not None:
                raise HttpRequestException(
                    f"SOAP Fault: {ET.tostring(fault, encoding='unicode')}"
                )
        fault = body.find("Fault")
        if fault is not None:
            raise HttpRequestException(
                f"SOAP Fault: {ET.tostring(fault, encoding='unicode')}"
            )

        children = list(body)
        return children[0] if len(children) == 1 else body
