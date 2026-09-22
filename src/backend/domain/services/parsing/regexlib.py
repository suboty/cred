import datetime

from asyncpg import TooManyRowsError

from logger import logger
from domain.services import SOAPMixin
from domain.entities.sources.regexlib import RegexLib


class RegexLibParser(SOAPMixin):
    def __init__(
            self,
            wsdl_link: str = 'https://regexlib.com/WebServices.asmx?wsdl'
    ):
        super().__init__(
            wsdl_link=wsdl_link
        )
        self.batch_count = 0
        self.founded_regexes = set()

    async def get_batch_regexes(
            self,
            batch_size: int = 500,
            is_need_all: bool = False
    ) -> list[RegexLib]:
        try:
            if is_need_all:
                batch = await self.soap_request(
                    0,
                    soap_action='ListAllAsXml'
                )
            else:
                batch = await self.soap_request(
                    (self.batch_count + 1) * batch_size,
                    soap_action='ListAllAsXml'
                )
            result = [
                RegexLib(
                    source_id=x.Id,
                    title=x.Title,
                    pattern=x.Pattern,
                    matching_text=x.MatchingText,
                    non_matching_text=x.NonMatchingText,
                    description=x.Description,
                    is_dirty=x.IsDirty,
                    author_name=x.AuthorName,
                    rating=x.Rating,
                    source_date_modified=x.DateModified,
                    created_at=datetime.datetime.now(),
                    updated_at=datetime.datetime.now()
                )
                for x in batch
            ]
            return result
        except Exception as e:
            logger.error(f"Error while RegexLib parser work: {str(e)[:500]}")
            raise e

    async def parse(
            self,
            batch_limit: int = 10,
            batch_size: int = 500,
            is_need_all: bool = False
    ) -> list:
        if batch_limit > 100:
            raise TooManyRowsError
        while self.batch_count != batch_limit:
            regexes = await self.get_batch_regexes(batch_size, is_need_all)
            for regex in regexes:
                self.founded_regexes.add(regex)
            self.batch_count += 1
        return list(self.founded_regexes)


if __name__ == '__main__':
    import asyncio

    async def main():
        a = RegexLibParser()
        res = await a.parse(
            batch_limit=1,
            batch_size=10
        )
        assert len(res) == 10

    asyncio.run(main())
