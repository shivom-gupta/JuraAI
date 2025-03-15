import os
import re
import asyncio
from crawl4ai import AsyncWebCrawler

output_dir = "data/bgb"
os.makedirs(output_dir, exist_ok=True)

base_url = "https://bgb.kommentar.de/"

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()

async def scrape_bgb():
    async with AsyncWebCrawler() as crawler:
        main_page_result = await crawler.arun(url=base_url)
        html_content = main_page_result.html
        section_pattern = r'<a href="(/[^"]+)">([^<]+)</a>'
        sections = re.findall(section_pattern, html_content)
        
        if not sections:
            print("No sections found. Please check the selector or website structure.")
            return

        for relative_url, section_name in sections:
            full_url = f"{base_url.rstrip('/')}{relative_url}"
            filename = sanitize_filename(section_name) + ".md"

            try:
                section_result = await crawler.arun(url=full_url)
                content = section_result.markdown
                
                output_path = os.path.join(output_dir, filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)
                
                print(f"Saved: {section_name} -> {output_path}")
            except Exception as e:
                print(f"Failed to fetch or save section {section_name} ({full_url}): {e}")

if __name__ == "__main__":
    asyncio.run(scrape_bgb())
