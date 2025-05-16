DOMAIN NAME GENERATOR - SETUP GUIDE
===================================

This script generates company name ideas and checks domain availability automatically.
Follow these steps to set up and run the script:

1. INSTALL REQUIRED PACKAGES:
   uv sync

2. CREATE A .env FILE:
   Create a file named '.env' in the same directory as this script with:
   OPENAI_API_KEY=your_openai_api_key_here
   
   You can get an API key from: https://platform.openai.com/api-keys

3. OUTPUT FILES:
   The script saves results to two files in the current directory:
   - available_domains.txt (domains that are available for registration)
   - checked_domains.txt (all domains that have been checked)
   
   You can modify these paths below if needed (search for AVAILABLE_DOMAINS_FILE and CHECKED_DOMAINS_FILE).

4. RUN THE SCRIPT:
   Basic usage:
   python AutoDomainGenerator.py --prompt "your company description"
   
   Interactive mode:
   python AutoDomainGenerator.py --interactive
   
   Clear history and start fresh:
   python AutoDomainGenerator.py --prompt "your company description" --clear-history

5. ADDITIONAL OPTIONS:
   --num 100                # Generate 100 names instead of default 50
   --min-delay 0.1          # Shorter delay between requests (faster but may hit rate limits)
   --max-delay 0.3          # Maximum initial delay
   --backoff-max 5.0        # Maximum delay during backoff
   --retries 3              # Number of retries for failed requests
   --model gpt-3.5-turbo    # Use a different OpenAI model (default is gpt-4o)

TROUBLESHOOTING:
- If you get many connection errors, try increasing delays with --min-delay and --max-delay
- If OpenAI API calls fail, check your API key and internet connection
- If the script seems slow, it's by design to avoid rate limiting from WHOIS servers
