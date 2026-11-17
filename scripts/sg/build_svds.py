"""
build_svds.py

Build video SVDs for one session.

Author: Stellina X. Ao
Created: 2025-03-26 # again, definitely created before, but lost the record
Last Modified: 2026-03-26
Python Version: 3.11.14
"""

from sg.svds import build_video_svd_data
from core.data import default_sessions

# build_video_svd_data(subj_id="MM012", sess_idx=5)
build_video_svd_data(subj_id="MR94", sess_id=default_sessions["MR94"])
