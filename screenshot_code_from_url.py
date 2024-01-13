import asyncio
import nest_asyncio
from pyppeteer import launch

nest_asyncio.apply()

async def take_screenshot():
    try:
        # Increase the timeout for browser launch (in milliseconds)
        browser = await launch(args=['--no-sandbox', '--disable-dev-shm-usage'], timeout=30000)
        page = await browser.newPage()

        # Navigate to the website
        await page.goto('https://www.amazon.com', {'waitUntil': 'networkidle0'})

        # Wait for 2 seconds using asyncio.sleep
        await asyncio.sleep(2)

        # Take a full-page screenshot with higher quality (adjust the quality value as needed)
        screenshot_path = 'screenshot.png'  # Replace with your desired file path
        await page.screenshot({'path': screenshot_path, 'fullPage': True, 'quality': 100})

        # Close the browser
        await browser.close()

    except Exception as e:
        print(f"Error during browser launch: {e}")

# Run the asyncio event loop
asyncio.get_event_loop().run_until_complete(take_screenshot())