    """Parse Ea value from string, extracting the first number."""
    if not val or val.strip() == '':
        return None
    val = val.strip()
    # Handle "109 +/- 1" format
    if '\u00b1' in val:
        val = val.split('\u00b1')[0].strip()
    # Handle "~111.49" format
    if '~' in val:
        val = val[1:].strip()
    try:
        parts = val.split()
        if parts:
            return float(parts[0])
    except (ValueError, IndexError):
        pass
    return None


