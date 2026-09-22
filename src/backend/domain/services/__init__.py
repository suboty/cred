import asyncio
import ssl
import json
import xml.etree.ElementTree as ET # noqa
from enum import Enum
from typing import Any
import logging

import zeep
import aiohttp

from logger import logger


class HttpRequestException(Exception):
    ...


class SoapRequestException(Exception):
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
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)[:500]}..."
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise HttpRequestException(f"Full attempts: {str(e)[:500]}") from e
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


class SOAPMixin:
    def __init__(
            self,
            *args,
            wsdl_link: str,
            max_retries: int = 3,
            retry_delay: int = 1,
            timeout: int = 30,
            **kwargs,
    ):
        log = logging.getLogger('zeep')
        log.handlers.clear()
        log.propagate = False
        log.disabled = False

        self.client = zeep.Client(wsdl=wsdl_link)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

    async def soap_request(
            self,
            *args,
            soap_action: str,
    ) -> Any:
        for attempt in range(self.max_retries):
            try:
                service = getattr(self.client.service, soap_action)
                return service(*args)
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)[:500]}..."
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise SoapRequestException(f"Full attempts: {str(e)[:500]}...") from e
        return None
