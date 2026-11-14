"""
Beetles AI - Lead Harvester & Drafter
Runs via GitHub Actions to queue cold leads.
"""

import json
import os

NEW_LEADS = [
    {
        "name": "Shannon L. Strong",
        "email": "shannonstronglawoffice@gmail.com",
        "hook": "CASA background in Travis County",
        "subject": "quick question about disputed screenshots in Travis County",
        "body": "Shannon,\n\nWith your CASA background in Travis County, you know how dangerous it is when an opposing party submits a doctored text screenshot during a temporary order hearing.\n\nWe built Beetles AI to analyze compression footprints and metadata in 30 seconds, giving you court-ready probability report cards.\n\nCould I send over a 1-page sample report card to keep on file for trial exhibits?\n\nMaryam Rashid\nBeetles AI Forensics"
    },
    {
        "name": "Johnny J. Urrutia",
        "email": "jurrutialaw@gmail.com",
        "hook": "Marine Corps background",
        "subject": "quick question re: doctored voice notes in Austin hearings",
        "body": "Johnny,\n\nIn South Austin criminal defense and high-conflict custody cases, hostile parties are increasingly using AI voice clones and edited text threads.\n\nBeetles AI verifies disputed audio clips and screenshots in 30 seconds, giving you an objective court-calibrated probability score.\n\nCan I email you a quick 1-page sample report card to keep on file for trial exhibits?\n\nMaryam Rashid\nBeetles AI Forensics"
    }
]

def update_queue():
    queue_file = "lead_queue.json"
    existing = []
    if os.path.exists(queue_file):
        with open(queue_file, "r") as f:
            try:
                existing = json.load(f)
            except Exception:
                existing = []
    
    existing_emails = {item.get("email") for item in existing}
    for lead in NEW_LEADS:
        if lead["email"] not in existing_emails:
            existing.append(lead)

    with open(queue_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Queue updated. Total leads in queue: {len(existing)}")

if __name__ == "__main__":
    update_queue()