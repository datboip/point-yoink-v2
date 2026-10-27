# SPDX-License-Identifier: AGPL-3.0-or-later
"""Object names for display; the saved optical keys remain compatible."""
LABELS={'near':'Small part','far':'Large part'}

def label(mode):return LABELS.get(str(mode).lower(),str(mode))
