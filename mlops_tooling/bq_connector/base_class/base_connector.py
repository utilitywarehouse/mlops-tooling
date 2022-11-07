from abc import ABC, abstractmethod
import jinja2
import pandas as pd


class BaseConnector(ABC):
    def __init__(self, path: str, credentials: dict):
        self._path = path
        self._credentials = credentials
        self._sql = self._set_path()

    @abstractmethod
    def _instantiate_connection(self):
        pass

    @abstractmethod
    def _set_path(self):
        return jinja2.Environment(loader=jinja2.FileSystemLoader(self._path))

    @abstractmethod
    def _read_query(self, sql_file: str, **kwargs) -> str:
        """
        Reads a SQL script and outputs a parsed version of the script.

        Args:
            file_name: a filename of the SQL script.

        Returns:
            query: parsed SQL script.
        """
        sql = self._sql.get_template(sql_file).render(kwargs)
        return sql

    @abstractmethod
    def query(self, sql_file: str) -> pd.DataFrame:
        pass
