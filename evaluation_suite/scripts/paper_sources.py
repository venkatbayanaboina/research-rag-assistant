from urllib.parse import urlparse

def get_pdf_url(url):
    """
    Return the direct PDF URL if we know how.
    Otherwise return None.
    """

    # -----------------------
    # arXiv
    # -----------------------
    if "arxiv.org/abs/" in url:
        return url.replace("/abs/", "/pdf/") + ".pdf"

    # Unknown source
    return None