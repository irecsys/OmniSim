"""
utils.dataset
=============
Dataset loader for OmniSim.

Reads three optional CSV files and exposes them as pandas DataFrames:

* ``items.csv``        — item metadata (required).
* ``users.csv``        — user demographic attributes (optional).
* ``interactions.csv`` — user–item ratings with optional timestamps (optional).

Column names are resolved from the configuration object so the same code
works across different domains (movies, fashion, e-commerce, …).
"""
import os
import pandas as pd

from utils.configurator import Config
from logging import getLogger


class Dataset(object):
    """CSV loader for OmniSim simulation data.

    Reads up to three CSV files from the dataset directory configured via
    ``data_path`` + ``dataset`` in the config:

    * ``file_items``        — item metadata (required; one row per item).
    * ``file_users``        — user demographic attributes (optional).
    * ``file_interactions`` — user–item ratings with optional timestamps (optional).

    Column names are resolved from the config so the same class works across
    different domains (movies, fashion, e-commerce, …).  After calling
    :meth:`load_data`, access data via ``self.df_items``, ``self.df_users``,
    and ``self.df_interactions``.
    """

    def __init__(self, config: Config) -> None:
        """Initialise the dataset loader and resolve column names from config.

        Args:
            config: Merged OmniSim configuration object (system + dataset YAML).
        """
        self.config = config
        self.config["path_inputs"] = config['data_path'] + config['dataset'] + '/'
        self.logger = getLogger()
        self.logger.info(f"Data path: {self.config['path_inputs']}")
        
        self.file_items = config['file_items']
        self.file_users = config['file_users']
        self.file_interactions = config['file_interactions']
        self.users_as_guest = config['users_as_guest']

        self.userid = config['col_userid'] 
        self.itemid = config['col_itemid']
        self.rating = config['col_rating']
        self.title = config['col_title']
        self.category = config['col_category']
        self.details = config['col_details']

    def _is_empty(self, val) -> bool:
        """Return True when a config value means "not provided".

        Treats Python ``None`` and the YAML literals ``~``, ``null``, and
        empty string as equivalent absence markers.
        """
        return val is None or str(val).strip().lower() in ('none', '~', 'null', '')

    def load_data(self) -> None:
        """Load items, users, and interactions CSVs into DataFrames.

        After this call the following attributes are available:

        * ``self.df_items``        — always a DataFrame (empty if no file configured).
        * ``self.df_users``        — DataFrame or ``None`` if not provided.
        * ``self.df_interactions`` — DataFrame or ``None`` if not provided or if
          ``users_as_guest`` is True (guest sessions need no history).

        The item-ID column is cast to ``str`` so it matches the string IDs
        returned by Elasticsearch.
        """
        # load item information (optional for user_guest strategy)
        if self._is_empty(self.file_items):
            self.df_items = pd.DataFrame()
            self.logger.info('No items file provided — items will be fetched from Elasticsearch.')
        else:
            csv_path = os.path.join(self.config["path_inputs"], self.file_items)
            self.df_items = pd.read_csv(csv_path, header=0, low_memory=False)
            self.df_items[self.itemid] = self.df_items[self.itemid].astype(str)

        # load user information (always, when provided)
        self.df_users = None
        if not self._is_empty(self.file_users):
            csv_path = os.path.join(self.config["path_inputs"], self.file_users)
            self.df_users = pd.read_csv(csv_path, header=0, low_memory=False)
            self.logger.info(f'Loaded {len(self.df_users)} users from {self.file_users}.')
        else:
            self.logger.info('User demographic information (optional) is not provided.')

        # load interactions (only when users_as_guest is False)
        self.df_interactions = None
        if self.users_as_guest is False:
            if self._is_empty(self.file_interactions):
                self.logger.info('No interactions file provided — user preference history unavailable.')
            else:
                csv_path = os.path.join(self.config["path_inputs"], self.file_interactions)
                self.df_interactions = pd.read_csv(csv_path, header=0, low_memory=False) 

        
        

        

   
