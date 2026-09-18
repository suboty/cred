import os
import datetime
import urllib.parse

from logger import logger
from domain.services import HTTPMixin, HttpMethod
from domain.entities.sources.regex101 import Regex101


class Regex101Parser(HTTPMixin):
    def __init__(
            self,
            regex101_baseurl: str = 'https://regex101.com/',
            library_endpoint: str = 'api/library',
            details_endpoint: str = 'api/library/details/',
            limit_pages: int | None = 10
    ):
        super().__init__()
        self.regex101_baseurl = regex101_baseurl
        self.library_endpoint = library_endpoint
        self.details_endpoint = details_endpoint
        self.page_count = 0
        self.current_cursor = os.getenv('CURRENT_CURSOR', '1')
        self.is_end = False
        self.founded_regexes = set()
        self.new_regexes_count = 0
        self.limit_pages = limit_pages

    async def parse(self):
        while not self.is_end:
            if self.limit_pages:
                if self.page_count >= self.limit_pages:
                    break
            await self.get_library_page()
        logger.info(
            f'Regex101 parsing is done\n'
            f'\tParsed pages: {self.page_count}\n'
            f'\tFounded regexes: {len(self.founded_regexes)}'
        )
        os.environ['CURRENT_CURSOR'] = self.current_cursor
        return self.founded_regexes

    async def get_library_page(self):
        try:
            response = await self.request(
                method=HttpMethod.GET,
                url=urllib.parse.urljoin(
                    self.regex101_baseurl,
                    self.library_endpoint
                ),
                url_params={'cursor': self.current_cursor}
            )
            if response.get('hasMore'):
                self.current_cursor = response.get('nextCursor')
                self.page_count += 1

                for regex_meta in response.get('data'):
                    permalink = regex_meta.get('permalinkFragment')
                    regex = await self.get_regex_by_permalink(permalink)
                    self.founded_regexes.add(
                        Regex101(
                            permalink=permalink,
                            regex=regex.get('regex'),
                            flags=regex.get('flags'),
                            delimiter=regex.get('delimiter'),
                            dialect=regex.get('flavor'),
                            title=regex.get('title'),
                            description=regex.get('description'),
                            created_at=datetime.datetime.now(),
                            updated_at=datetime.datetime.now()
                        )
                    )

            else:
                self.is_end = True
        finally:
            await self.close()

    async def get_regex_by_permalink(self, permalink: str):
        try:
            return await self.request(
                method=HttpMethod.GET,
                url=urllib.parse.urljoin(
                    urllib.parse.urljoin(
                        self.regex101_baseurl,
                        self.details_endpoint,
                    ),
                    permalink
                )
            )
        finally:
            await self.close()


if __name__ == '__main__':
    import asyncio

    async def main():
        a = Regex101Parser(limit_pages=1)
        res = await a.parse()
        print(len(res))
        print(res)

    asyncio.run(main())
