from astroquery.mast import Catalogs

__all__ = ["get_tic_info"]


def get_tic_info(TIC_ID):
    """
    Get the information about this TIC ID from the TESS Input Catalog
    at MAST.

    Parameters
    ----------

    TIC_ID : int
        9 or 10 digit TIC ID number.

    Returns
    -------

    `astropy.table.Table`
        Astropy table withinformation about the TIC object.

    """
    catalog_data = Catalogs.query_criteria(catalog="Tic", ID=TIC_ID)
    return catalog_data
