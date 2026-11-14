"""
Beetles AI - Dynamic Lead & $100M Offer Engine v3.0
Generates court-specific, zero-risk outreach for Texas trial lawyers.
Tracks status ('queued' / 'sent') to prevent duplicates permanently.
"""

import json
import os

QUEUE_FILE = "lead_queue.json"


def build_100m_offer_email(attorney_name: str, firm_name: str, city: str, county: str, niche_hook: str) -> dict:
    """Dynamically generates $100M Hormozi Offer copy tailored to any Texas attorney."""
    subject = f"quick question about disputed evidence in {county} County"
    
    body = f"""{attorney_name},

{niche_hook}

In high-conflict family and trial litigation in {county} County, opposing counsel frequently attempts to introduce manipulated text screenshots, cropped social media posts, or altered voice notes.

Proving digital evidence is altered usually takes $3,500+ in expert witness fees and 3 weeks of discovery delay—far too slow for emergency temporary order hearings.

We built Beetles AI to solve this in 30 seconds. Our forensic engine analyzes JPEG compression layers, Fourier spectral patterns, and EXIF metadata into a court-ready probability report card.

The Offer: If {firm_name} has an active case right now with a disputed screenshot or voice note, email it over. We will run a complete forensic audit on your first exhibit free of charge so you can test the analysis live in court.

If it does not give you immediate leverage with the judge, you owe us zero.

Would you be against me sending over a 1-page sample forensic report card to see how the audit works?

Maryam Rashid
Beetles AI Forensics"""

    return {
        "subject": subject,
        "body": body
    }


def add_leads_to_queue(raw_leads: list):
    """
    Appends raw attorney contacts into lead_queue.json.
    Ensures no duplicate emails exist.
    """
    existing_queue = []
    
    if os.path.exists(QUEUE_FILE):
        try:
            with open(QUEUE_FILE, "r", encoding="utf-8") as f:
                existing_queue = json.load(f)
        except Exception:
            existing_queue = []

    existing_emails = {item.get("email", "").lower().strip() for item in existing_queue}
    added_count = 0

    for lead in raw_leads:
        email = lead.get("email", "").lower().strip()
        if not email or email in existing_emails:
            continue

        email_data = build_100m_offer_email(
            attorney_name=lead.get("attorney_name", "Counsel"),
            firm_name=lead.get("firm_name", "your firm"),
            city=lead.get("city", "Texas"),
            county=lead.get("county", "Texas"),
            niche_hook=lead.get("niche_hook", "When clients face disputed digital evidence, unverified screenshots can derail a hearing before discovery opens.")
        )

        record = {
            "id": len(existing_queue) + 1,
            "attorney_name": lead.get("attorney_name"),
            "email": email,
            "firm_name": lead.get("firm_name"),
            "city": lead.get("city"),
            "county": lead.get("county"),
            "subject": email_data["subject"],
            "body": email_data["body"],
            "status": "queued"  # 'queued' or 'sent'
        }

        existing_queue.append(record)
        existing_emails.add(email)
        added_count += 1

    with open(QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_queue, f, indent=2, ensure_ascii=False)

    print(f"✅ Success: Added {added_count} new leads to {QUEUE_FILE}. Total leads: {len(existing_queue)}")


if __name__ == "__main__":
    # Add new target batches here as you find them (or via automated scraper)
    BATCH = [
        {
            "attorney_name": "Donn Fullenweider",
            "firm_name": "Fullenweider Wilite LLP",
            "email": "info@fullenweider.com",
            "city": "Houston",
            "county": "Harris",
            "niche_hook": "In high-net-worth family litigation in Harris County, opposing parties frequently attempt to introduce manipulated text screenshots or fabricated financial records to sway emergency rulings."
        },
        {
            "attorney_name": "Lindsey Obenhaus",
            "firm_name": "Goranson Bain Ausley",
            "email": "contact@gbafamilylaw.com",
            "city": "Dallas",
            "county": "Dallas",
            "niche_hook": "In complex Dallas custody disputes, unverified text screenshots and cropped audio notes often enter the record before discovery even opens."
        },
        {
            "attorney_name": "Ike Vanden Eykel",
            "firm_name": "KoonsFuller Family Law",
            "email": "dallasinfo@koonsfuller.com",
            "city": "Dallas",
            "county": "Dallas",
            "niche_hook": "Handling high-conflict divorce cases in Dallas requires shutting down unverified digital claims before they influence a judge's ruling."
        }
    ]
    
    add_leads_to_queue(BATCH)