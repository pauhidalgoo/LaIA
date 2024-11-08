import asyncio
import nest_asyncio
nest_asyncio.apply()

import asyncio
from crawl4ai import AsyncWebCrawler


async def clean_content():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://web.gencat.cat/ca/tramits/tramits-temes?filtreResp=p",

        )
        full_markdown_length = len(result.markdown)
        fit_markdown_length = len(result.fit_markdown)
        print(f"Full Markdown Length: {full_markdown_length}")
        print(f"Fit Markdown Length: {fit_markdown_length}")
        print(result.markdown)


asyncio.run(clean_content())