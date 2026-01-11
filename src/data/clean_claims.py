import csv
import re

def extract_first_claim(main_claim):
    """Extract only the first substantive claim from the main_claim text."""
    if not main_claim or main_claim.strip() == "":
        return main_claim
    
    # Find all numbered claims in the text
    # Pattern matches claim numbers like "1 .", "23 .", "1-22 .", etc.
    claim_pattern = r'(\d+(?:\s*-\s*\d+)?)\s*\.\s*'
    
    # Find all claim positions
    matches = list(re.finditer(claim_pattern, main_claim))
    
    if not matches:
        # No numbered claims found
        return main_claim
    
    # Check each claim to find the first non-canceled one
    for i, match in enumerate(matches):
        claim_start = match.start()
        claim_number_end = match.end()
        
        # Determine where this claim ends (either at the next claim or end of text)
        if i + 1 < len(matches):
            claim_end = matches[i + 1].start()
        else:
            claim_end = len(main_claim)
        
        # Extract the claim content
        claim_content = main_claim[claim_number_end:claim_end].strip()
        
        # Check if this claim is canceled or empty
        # Look for patterns like "(canceled)", "(cancelled)", or very short content
        if re.search(r'\(cancel(?:ed|led)\)', claim_content, re.IGNORECASE):
            continue  # Skip canceled claims
        
        # Check if the claim has substantial content (more than just whitespace/punctuation)
        # A real claim should have at least some alphabetic characters
        if len(re.findall(r'[a-zA-Z]', claim_content)) < 10:
            continue  # Skip claims with minimal content
        
        # Found a valid claim! Extract it (including the claim number)
        full_claim = main_claim[claim_start:claim_end].strip()
        return full_claim
    
    # If all claims were canceled, return the original text
    return main_claim

# Read the CSV file
input_file = 'patents_5_cpc_classes.csv'
output_file = 'patents_5_cpc_classes_cleaned.csv'

rows = []
with open(input_file, 'r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    
    for row in reader:
        # Extract only the first claim
        row['main_claim'] = extract_first_claim(row['main_claim'])
        rows.append(row)

# Write the cleaned data to a new CSV file
with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"✓ Processed {len(rows)} rows")
print(f"✓ Cleaned CSV saved to: {output_file}")
