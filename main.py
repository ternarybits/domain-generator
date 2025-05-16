import argparse
import json
import os
import random
import time
from typing import Dict, List, Set, Tuple

import openai
import whois
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# File to store available domains (in the current directory)
AVAILABLE_DOMAINS_FILE = "available_domains.txt"
# File to store checked domains (both available and taken)
CHECKED_DOMAINS_FILE = "checked_domains.txt"


def load_checked_domains() -> Set[str]:
    """Load all previously checked domains to avoid duplicates"""
    checked_domains = set()

    if os.path.exists(CHECKED_DOMAINS_FILE):
        with open(CHECKED_DOMAINS_FILE, "r") as f:
            for line in f:
                domain = (
                    line.strip().lower()
                )  # Convert to lowercase for case-insensitive comparison
                if domain:
                    checked_domains.add(domain)

    return checked_domains


def save_checked_domains(domains: List[str]) -> None:
    """Save checked domains to a file, avoiding duplicates"""
    # Load existing checked domains
    checked_domains = load_checked_domains()

    # Filter out domains that already exist in the file (case insensitive)
    new_domains = [
        domain for domain in domains if domain.lower() not in checked_domains
    ]

    if not new_domains:
        return

    # Append new domains to the file
    with open(CHECKED_DOMAINS_FILE, "a") as f:
        for domain in new_domains:
            f.write(f"{domain}\n")

    print(f"Added {len(new_domains)} new domains to checked domains list")


def filter_already_checked_domains(
    company_names: List[str], checked_domains: Set[str], tlds: List[str]
) -> List[str]:
    """
    Filter out company names that would result in domain names (for any specified TLD)
    that have already been checked.

    Args:
        company_names: List of company names to filter
        checked_domains: Set of domain names that have already been checked (e.g., {'example.ai', 'test.com'})
        tlds: List of TLDs to consider for each company name (e.g., ['.ai', '.com'])

    Returns:
        List of company names that would result in new domain names for all specified TLDs.
    """
    filtered_names = []
    filtered_out_count = 0

    for name in company_names:
        # New logic: filter out if any TLD variation is already checked
        should_filter_out = False
        normalized_name_part = name.lower().replace(" ", "")

        if (
            not normalized_name_part
        ):  # Handle cases where name might be empty or just spaces
            filtered_out_count += 1
            continue  # Skip this name

        for tld in tlds:
            domain_name_to_check = f"{normalized_name_part}{tld}"
            if domain_name_to_check.lower() in checked_domains:
                should_filter_out = True
                break  # Found one checked domain, no need to check other TLDs for this name

        if not should_filter_out:
            filtered_names.append(name)
        else:
            filtered_out_count += 1

    if filtered_out_count > 0:
        print(
            f"Filtered out {filtered_out_count} company names that would result in already checked domains"
        )

    return filtered_names


def generate_company_names(
    prompt: str,
    num_names: int = 50,
    model: str = "gpt-4o",
    checked_domains: Set[str] | None = None,
    tlds: List[str] | None = None,
) -> List[str]:
    """
    Generate company name ideas using OpenAI's API based on a prompt.

    Args:
        prompt: Description of the type of company names to generate
        num_names: Number of names to generate
        model: OpenAI model to use
        checked_domains: Set of domain names that have already been checked
        tlds: List of TLDs the names should be suitable for (e.g., ['.ai', '.com'])

    Returns:
        List of company name ideas
    """
    if tlds is None:
        tlds = [".ai"]

    try:
        client = openai.OpenAI()

        #  - consider modifying words with a suffix or prefix to create a new word, similar to something like 'Partiful'
        # - consider integrating phtoography terms like portrait, photo, snap, lens, aperture, film, focus, exposure, obscura, darkroom, bokeh
        # - consider social terms like crew, group, squad, friends,

        system_message = """You are a creative naming consultant for gen-z hip startups. 
        Generate unique, memorable company names based on the user's description.
        Name should be catchy and gen-z hip. 
        explore using a few letters without vowels like common startups.
        name should be 8 letters or less but can mash together multiple words into one.
        Return ONLY a JSON array of strings with no additional text or explanation."""

        # Create a sample of checked domains to show the model (limit to 20 to avoid token issues)
        checked_domains_sample = []
        if checked_domains:
            checked_domains_sample = (
                list(checked_domains)[-20:]
                if len(checked_domains) > 20
                else list(checked_domains)
            )

        # Include checked domains in the prompt if available
        domains_to_avoid = ""
        if checked_domains_sample:
            domains_to_avoid = f"""
        IMPORTANT: The following domain names have already been checked and are NOT available:
        {", ".join(checked_domains_sample)}
        
        DO NOT generate names that would result in similar domain names. Be creative and generate completely different options.
        """

        user_message = f"""Generate {num_names} unique company names for: {prompt}
        
        Important guidelines:
        - Names should be 1 word maximum
        - Memorable and distinctive
        - Good potential as domain names with TLDs: {", ".join(tlds)}
        - No hyphens or special characters
        {domains_to_avoid}
        Return ONLY a JSON array of strings."""

        # Try with explicit JSON format first
        try:
            response = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.90,
            )

            # Check if we got a valid response
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from OpenAI API")

            # Try to parse the JSON
            names_data = json.loads(content)

            # Extract the array of names (handle both possible formats)
            if "names" in names_data:
                return names_data["names"]
            else:
                # Assume the entire object is the array
                return list(names_data.values())[0]

        except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
            print(
                f"Error with JSON response format: {str(e)}. Trying without explicit format..."
            )

            # Fallback to standard completion without JSON format specification
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                        + " Format your response as a valid JSON array.",
                    },
                    {
                        "role": "user",
                        "content": user_message
                        + "\n\nIMPORTANT: Your response MUST be a valid JSON array of strings only.",
                    },
                ],
                temperature=0.97,
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from OpenAI API")

            # Try to extract just the JSON part from the response
            try:
                # Look for JSON array pattern
                import re

                json_match = re.search(r'\[\s*"[^"]*"(?:\s*,\s*"[^"]*")*\s*\]', content)
                if json_match:
                    names = json.loads(json_match.group(0))
                    return names

                # If that fails, try to find any JSON object
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    names_data = json.loads(json_match.group(0))
                    if "names" in names_data:
                        return names_data["names"]
                    else:
                        # Try to find any array in the JSON
                        for value in names_data.values():
                            if isinstance(value, list):
                                return value

                # Last resort: manually parse the response for anything that looks like company names
                lines = content.split("\n")
                names = []
                for line in lines:
                    # Look for lines that might contain a name (e.g., "1. CompanyName")
                    name_match = re.search(
                        r"(?:^|\s)(?:\d+\.\s*)?([A-Z][a-zA-Z0-9]*(?:\s[A-Z][a-zA-Z0-9]*)?)",
                        line,
                    )
                    if (
                        name_match and len(name_match.group(1)) > 2
                    ):  # Ensure name is at least 3 chars
                        names.append(name_match.group(1))

                if names:
                    return names

                # If all else fails, raise an error
                raise ValueError("Could not extract company names from response")

            except Exception as inner_e:
                print(f"Error extracting names from response: {str(inner_e)}")
                print(f"Raw response: {content}")
                raise

    except Exception as e:
        print(f"Error generating company names: {str(e)}")
        if "content" in locals():
            print(f"Response content: {content}")
        return []


def check_domain_availability(
    company_names: List[str],
    initial_delay_range: Tuple[float, float] = (0.2, 0.5),
    max_delay: float = 10.0,
    max_retries: int = 5,
    tlds: List[str] | None = None,
) -> Dict[str, bool]:
    """
    Check domain availability for a list of company names across specified TLDs
    with adaptive rate limiting and exponential backoff.

    Args:
        company_names: List of company names to check
        initial_delay_range: Tuple of (min, max) seconds to wait between initial requests
        max_delay: Maximum delay in seconds for backoff
        max_retries: Maximum number of retries on connection errors
        tlds: List of TLDs to check for each company name (e.g., [".ai", ".com"])
    """
    if tlds is None:
        tlds = [".ai"]

    available_domains = {}
    total = len(company_names)

    # Track consecutive failures to implement adaptive delays
    consecutive_failures = 0
    current_delay_range = initial_delay_range

    # Keep track of all domains checked in this specific run
    all_checked_domains_in_run: List[str] = []

    # Prepare the list of actual domain names to check by combining company names and TLDs
    domains_to_check_list: List[str] = []
    for name in company_names:
        normalized_name_part = name.lower().replace(" ", "")
        if (
            not normalized_name_part
        ):  # Skip if the name part is empty after normalization
            continue
        for tld in tlds:
            domains_to_check_list.append(f"{normalized_name_part}{tld}")

    total_domains_to_check = len(domains_to_check_list)

    if total_domains_to_check == 0:
        print("No valid domains to check after processing company names and TLDs.")
        # Still call save_checked_domains, it handles empty lists correctly (no-op)
        save_checked_domains([])
        return {}

    # Iterate over the fully formed domain names
    for idx, domain_name in enumerate(domains_to_check_list, 1):
        success = False

        print(f"Checking {idx}/{total_domains_to_check}: {domain_name}")
        all_checked_domains_in_run.append(domain_name)

        for attempt in range(max_retries):
            try:
                domain = whois.whois(domain_name)

                # Check if the domain exists based on the error message
                if isinstance(
                    domain.status, (type(None), str)
                ) and "No match for" in str(domain):
                    available_domains[domain_name] = True
                else:
                    available_domains[domain_name] = False

                success = True
                consecutive_failures = 0  # Reset failure counter on success
                break

            except Exception as e:
                if "No match for" in str(e):
                    available_domains[domain_name] = True
                    success = True
                    consecutive_failures = 0  # Reset failure counter on success
                    break
                elif "Connection reset by peer" in str(e) or "timed out" in str(e):
                    consecutive_failures += 1

                    # Calculate backoff delay based on attempt number (exponential backoff)
                    backoff_factor = min(2**attempt, max_delay / current_delay_range[1])
                    wait_time = random.uniform(
                        current_delay_range[0] * backoff_factor,
                        current_delay_range[1] * backoff_factor,
                    )

                    if attempt < max_retries - 1:  # if we still have retries left
                        print(
                            f"  Connection error, retry {attempt + 1}/{max_retries} in {wait_time:.1f} seconds..."
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        available_domains[domain_name] = (
                            f"Connection error after {max_retries} attempts"
                        )
                else:
                    available_domains[domain_name] = f"Error checking: {str(e)}"

        if not success:
            print(f"  Failed to check {domain_name} after {max_retries} attempts")

        # Adaptive delay between requests based on consecutive failures
        if idx < total_domains_to_check:  # Don't delay after the last request
            # Increase delay if we're experiencing failures
            if consecutive_failures > 2:
                # Gradually increase delay range up to max_delay
                adjusted_min = min(current_delay_range[0] * 1.5, max_delay / 2)
                adjusted_max = min(current_delay_range[1] * 1.5, max_delay)
                current_delay_range = (adjusted_min, adjusted_max)
                print(
                    f"  Increasing delay due to consecutive failures: {current_delay_range[0]:.1f}-{current_delay_range[1]:.1f}s"
                )
            elif (
                consecutive_failures == 0
                and current_delay_range[0] > initial_delay_range[0]
            ):
                # Gradually decrease delay back toward initial range after successful requests
                adjusted_min = max(current_delay_range[0] * 0.8, initial_delay_range[0])
                adjusted_max = max(current_delay_range[1] * 0.8, initial_delay_range[1])
                current_delay_range = (adjusted_min, adjusted_max)

            delay = random.uniform(current_delay_range[0], current_delay_range[1])
            time.sleep(delay)

    # Save all checked domains to the file
    save_checked_domains(all_checked_domains_in_run)

    return available_domains


def load_existing_domains() -> Set[str]:
    """Load previously saved available domains to avoid duplicates"""
    existing_domains = set()

    if os.path.exists(AVAILABLE_DOMAINS_FILE):
        with open(AVAILABLE_DOMAINS_FILE, "r") as f:
            for line in f:
                domain = (
                    line.strip().lower()
                )  # Convert to lowercase for case-insensitive comparison
                if domain:
                    existing_domains.add(domain)

    return existing_domains


def save_available_domains(domains: List[str], existing_domains: Set[str]) -> None:
    """Save available domains to a file, avoiding duplicates"""
    # Filter out domains that already exist in the file (case insensitive)
    new_domains = [
        domain for domain in domains if domain.lower() not in existing_domains
    ]

    if not new_domains:
        print("No new available domains to save.")
        return

    # Append new domains to the file
    with open(AVAILABLE_DOMAINS_FILE, "a") as f:
        for domain in new_domains:
            f.write(f"{domain}\n")

    print(f"Saved {len(new_domains)} new available domains to {AVAILABLE_DOMAINS_FILE}")


def auto_generate_and_check(
    prompt: str,
    num_names: int = 50,
    model: str = "gpt-4o",
    initial_delay_range: Tuple[float, float] = (0.2, 0.5),
    max_delay: float = 10.0,
    max_retries: int = 5,
    tlds: List[str] | None = None,
) -> None:
    """
    Automatically generate company names and check domain availability in one go.

    Args:
        prompt: Description of the type of company names to generate
        num_names: Number of names to generate
        model: OpenAI model to use
        initial_delay_range: Tuple of (min, max) seconds to wait between initial requests
        max_delay: Maximum delay in seconds for backoff
        max_retries: Maximum number of retries on connection errors
        tlds: List of TLDs to check (e.g., [".ai", ".com"])
    """
    if tlds is None:
        tlds = [
            ".ai"
        ]  # Default to .ai if not provided, though main should always provide

    # Load existing domains to avoid duplicates
    existing_domains = load_existing_domains()
    print(
        f"Loaded {len(existing_domains)} existing domains from {AVAILABLE_DOMAINS_FILE}"
    )

    # Load previously checked domains to avoid generating similar names
    checked_domains = load_checked_domains()
    print(f"Loaded {len(checked_domains)} previously checked domains")

    # Generate company names
    print(
        f"\nGenerating {num_names} unique company names for: {prompt} (TLDs: {', '.join(tlds)})"
    )
    company_names = generate_company_names(
        prompt, num_names, model, checked_domains, tlds
    )

    if not company_names:
        print("Failed to generate company names. Please try again.")
        return

    # Filter out company names that would result in already checked domains
    original_count = len(company_names)
    company_names = filter_already_checked_domains(company_names, checked_domains, tlds)

    if len(company_names) == 0:
        print(
            "All generated domains have already been checked. Please try again with a different prompt."
        )
        return

    if len(company_names) < original_count:
        print(
            f"Filtered out {original_count - len(company_names)} already checked domains. Proceeding with {len(company_names)} unique names."
        )

    print(f"\nGenerated {len(company_names)} company names:")
    for i, name in enumerate(company_names, 1):
        print(f"{i}. {name}")

    # Check domain availability
    print(
        f"\nStarting domain checks for {len(company_names)} names across TLDs: {', '.join(tlds)}..."
    )
    print("This may take a few minutes due to rate limiting...")
    print("-" * 50)

    results = check_domain_availability(
        company_names,
        initial_delay_range=initial_delay_range,
        max_delay=max_delay,
        max_retries=max_retries,
        tlds=tlds,
    )

    # Process results to determine availability at the company name level
    company_availability_summary: Dict[str, Dict[str, List[str]]] = {}
    all_actually_available_fqdns: List[str] = []  # For saving to file

    for name in company_names:  # Iterate over the unique base company names
        normalized_name_part = name.lower().replace(" ", "")
        if not normalized_name_part:  # Should have been filtered by generate_company_names or filter_already_checked
            continue

        company_availability_summary[name] = {"available": [], "taken": [], "error": []}

        for tld in tlds:
            domain_to_check = f"{normalized_name_part}{tld}"
            status = results.get(domain_to_check)  # Get status for this specific FQDN

            if status is True:
                company_availability_summary[name]["available"].append(domain_to_check)
                all_actually_available_fqdns.append(domain_to_check)
            elif status is False:
                company_availability_summary[name]["taken"].append(domain_to_check)
            elif isinstance(status, str):  # Error message string
                company_availability_summary[name]["error"].append(
                    f"{domain_to_check}: {status}"
                )
            # If status is None, it means the domain wasn't in results (e.g., check failed silently or skipped)
            # This case should be minimal if check_domain_availability processed all combinations.

    # Prepare display lists
    available_company_display_list = []
    unavailable_company_display_list = []

    for name, summary in company_availability_summary.items():
        if summary[
            "available"
        ]:  # Company name is available if at least one TLD is available
            available_details = ", ".join(summary["available"])

            other_statuses = []
            if summary["taken"]:
                other_statuses.append(f"taken in: {', '.join(summary['taken'])}")
            if summary["error"]:
                other_statuses.append(f"errors for: {', '.join(summary['error'])}")

            other_info_str = ""
            if other_statuses:
                other_info_str = f" (other TLDs - {'; '.join(other_statuses)})"

            available_company_display_list.append(
                f"  • {name} (available as: {available_details}){other_info_str}"
            )
        else:  # Not available in any specified TLD for this company name
            parts = []
            if summary["taken"]:
                parts.append(f"taken in: {', '.join(summary['taken'])}")
            if summary["error"]:
                parts.append(f"errors for: {', '.join(summary['error'])}")

            detail_str = "; ".join(parts)
            if not detail_str:
                # This case implies the name was processed but no TLDs resulted in True/False/Error string.
                # Could happen if all results were None for its TLDs, or if name had no TLDs (unlikely here).
                detail_str = "status unknown for all TLDs or no TLDs processed"

            unavailable_company_display_list.append(f"  • {name} ({detail_str})")

    # Print grouped results based on company name availability
    print("\nCompany Name Availability Results")
    print("=" * 50)

    print("\n✅ COMPANY NAMES AVAILABLE (in at least one specified TLD):")
    print("-" * 20)
    if available_company_display_list:
        for item in sorted(available_company_display_list):
            print(item)
    else:
        print("  None found")

    print("\n❌ COMPANY NAMES NOT AVAILABLE (in any specified TLD):")
    print("-" * 20)
    if unavailable_company_display_list:
        for item in sorted(unavailable_company_display_list):
            print(item)
    else:
        print("  None found (or all were available)")

    # Summary based on company names
    total_company_names_processed = len(company_names)
    num_companies_available = len(available_company_display_list)
    num_companies_unavailable = len(unavailable_company_display_list)

    print("\nSummary (based on Company Names):")
    print(f"Total company names processed: {total_company_names_processed}")
    print(f"Company names available in at least one TLD: {num_companies_available}")
    print(f"Company names not available in any TLD: {num_companies_unavailable}")

    # Save all *actually available FQDNs* to the file
    if all_actually_available_fqdns:
        print("\nSaving specific available domain names...")
        save_available_domains(all_actually_available_fqdns, existing_domains)
    else:
        print("\nNo new specific domain names found to be available for saving.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate company names and check domain availability"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="Description of the type of company names to generate",
    )
    parser.add_argument(
        "--num",
        "-n",
        type=int,
        default=50,
        help="Number of names to generate (default: 50)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        default=0.2,
        help="Minimum initial delay between requests in seconds (default: 0.2)",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=0.5,
        help="Maximum initial delay between requests in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--backoff-max",
        type=float,
        default=10.0,
        help="Maximum delay for backoff in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--retries",
        "-r",
        type=int,
        default=5,
        help="Maximum number of retries for failed requests (default: 5)",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--clear-history",
        action="store_true",
        help="Clear the history of checked domains",
    )
    parser.add_argument(
        "--tlds",
        type=str,
        nargs="+",  # Accepts one or more arguments
        default=[".ai"],
        help="List of TLDs to search (e.g., .com .io .ai). Default: .ai",
    )

    args = parser.parse_args()

    # Sanitize TLDs to ensure they start with a dot
    sanitized_tlds = []
    for tld in args.tlds:
        if not tld.startswith("."):
            sanitized_tlds.append(f".{tld}")
        else:
            sanitized_tlds.append(tld)
    args.tlds = list(set(sanitized_tlds))  # Remove duplicates and update args
    print(f"TLDs to search: {args.tlds}")

    # Handle clearing history if requested
    if args.clear_history:
        if os.path.exists(CHECKED_DOMAINS_FILE):
            os.remove(CHECKED_DOMAINS_FILE)
            print("Cleared checked domains history")

    if args.interactive:
        # Interactive mode
        existing_domains = load_existing_domains()
        print(
            f"Loaded {len(existing_domains)} existing domains from {AVAILABLE_DOMAINS_FILE}"
        )

        while True:
            # Get user input for the type of company names to generate
            prompt = input(
                "\nDescribe the type of company names to generate (or 'quit' to exit): "
            )

            if prompt.lower() in ("quit", "exit", "q"):
                break

            # Get number of names to generate
            try:
                num_names = int(
                    input(f"How many names to generate? [{args.num}]: ")
                    or str(args.num)
                )
            except ValueError:
                num_names = args.num

            auto_generate_and_check(
                prompt=prompt,
                num_names=num_names,
                model=args.model,
                initial_delay_range=(args.min_delay, args.max_delay),
                max_delay=args.backoff_max,
                max_retries=args.retries,
                tlds=args.tlds,  # Pass sanitized TLDs
            )
    else:
        # Non-interactive mode, requires prompt
        if not args.prompt:
            print("Error: --prompt is required in non-interactive mode")
            parser.print_help()
            return

        auto_generate_and_check(
            prompt=args.prompt,
            num_names=args.num,
            model=args.model,
            initial_delay_range=(args.min_delay, args.max_delay),
            max_delay=args.backoff_max,
            max_retries=args.retries,
            tlds=args.tlds,  # Pass sanitized TLDs
        )


if __name__ == "__main__":
    main()
