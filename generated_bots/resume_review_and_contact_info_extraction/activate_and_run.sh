#!/bin/bash
echo "Activating environment for resume_review_and_contact_info_extraction..."
source "generated_bots/resume_review_and_contact_info_extraction/venv/bin/activate"
echo "Environment activated! Python path: generated_bots/resume_review_and_contact_info_extraction/venv/bin/python"
echo ""
echo "To run the bot:"
echo "python resume_review_and_contact_info_extraction.py"
echo ""
exec "$SHELL"
