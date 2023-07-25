#!/bin/python3

import re
import argparse
from tabula import read_pdf
from functools import reduce, cached_property
from decimal import Decimal
from copy import deepcopy
from pathlib import Path
from itertools import zip_longest
import pandas

def list_diff(l1: list, l2: list) -> list[int]:
    return [i for i, (e1, e2) in enumerate(zip_longest(l1, l2)) if e1 != e2]

class RefactoredPDF:
    def __init__(self, filename: str):
        self.filename = filename
        self._read_pdf()
        self._clean_pdf()

    def _read_pdf(self):
        self.raw_dfs = read_pdf(
            self.filename,
            multiple_tables=True,
            lattice=True,
            pages="all",
            encoding='ISO-8859-1'
        )
        # self.split_count
        self._check_lines_count()
        self._refactor_columns_names()
        self.df = self._concat_columns()

    def _clean_pdf(self):
        self._drop_stats_line()
        self._split_resultats()
        self._normalize()
        self.rename_header()

    @cached_property
    def sheet_count(self) -> int:
        return len(self.raw_dfs)

    """
    Number of pages used by the pdf to represent a record
    """
    @cached_property
    def split_count(self) -> int:
        cols_len = [len(df.columns) for df in self.raw_dfs]
        split_count = cols_len.index(min(cols_len)) + 1
        group_pattern = cols_len[:split_count]
        # All splitted pages should have the same group pattern
        diff = list_diff((self.sheet_count // split_count) * group_pattern, cols_len)
        
        if diff:
            raise ValueError(
                f"Thoses pages have an inconsistant number of columns "
                f"compared to the first group ({group_pattern}) :\n {diff}")

        return split_count

    """
    Assert parsed pdf's lines have a coherent number of lines across
    splitted pages
    """
    def _check_lines_count(self) -> bool:
        lines_len = [len(d) for d in self.raw_dfs]
        for i in range(self.sheet_count, self.split_count):
            for length in lines_len[i:i+self.split_count]:
                if length != lines_len[0]:
                    raise ValueError(
                        f"Page n°{i} have an inconsistant number of lines : "
                        f"{length}, expected {lines_len[0]}")
        return True

    """
    Realign columns names, and rename student identifier columns.
    """
    def _refactor_columns_names(self):
        for df in self.raw_dfs:
            cols = list(df.columns)
            cols.insert(1, '')
            cols.pop()
            cols[0] = "NumEtudiant"
            df.columns = cols
            df.drop('', axis=1, inplace=True)
    
    """
    Concat pdf's pages of form [1A, 2B, 3C, 4A, 5B, 6C, ...]
    into [1A + 2B + 3C, 4A + 5B + 6C, ...], where '+' is a
    concatenation briging together each columns to create a single record.
    Then assemble all dataframes into a single one.
    """
    def _concat_columns(self):
        merged = []
        # print(len(self.df))
        # Merge columns that were splitted accros sheets
        for first_page in range(0, self.sheet_count, self.split_count):
            splitted_pages = [
                self.raw_dfs[i]
                for i in range(first_page, first_page + self.split_count, 1)
            ]
            # print(f"{first_page=}")
            merged.append(
                reduce(
                    lambda df1, df2: pandas.concat(
                        [df1, df2.drop(df2.columns[0], axis=1)],
                        axis=1
                    ),
                    splitted_pages
                )
            )
        # Merges all lines into one dataframe
        df = pandas.concat(merged, axis=0)
        df.reset_index(drop=True, inplace=True)

        return df

    """
    Remove statistics.
    """
    def _drop_stats_line(self):
        self.df.drop(self.df.tail(1).index, inplace=True)

    """
    Split "Résultat" column in two column "Résultat" and "Admission" 
    """
    def _split_resultats(self):
        resultats = self.df[self.df.columns[1]]
        
        # Split 'Résultat' column that contains 'admission' status 
        resultats = resultats.str.split('\r', expand=True).iloc[:, :2]
        
        # Force two columns if only one was found
        indexes = (0,1) if len(resultats.columns) == 2 else (1,0)
        resultats = resultats.reindex(labels=indexes, axis=1, fill_value=float('NaN'))
        
        resultats.columns = ["Résultat", "Admission"]
        self.df = pandas.concat(
            [self.df.iloc[:, 0:1], resultats, self.df.iloc[:, 2:]], axis=1
        )

    """
    Refactor cells to only contains cells's numbers. 
    """
    def _normalize(self):
        def extract_number(cell: str):
            if match := re.search('(\\d*\\.?\\d+)', str(cell)):
                return Decimal(match.group())
            return Decimal("NaN")

        left  = self.df.iloc[:, 0:2].applymap(extract_number)
        right = self.df.iloc[:, 3: ].applymap(extract_number)

        self.df = self.df.assign(**left).assign(**right)

    """
    Extract all UE codes from header
    """
    @cached_property
    def courses_ids(self):
        return [
            re.search("^([A-Z]|\\d){7}|$", field).group()
            for field in self.df.columns
        ]

    """
    Extract all UE names from header
    """
    @cached_property
    def courses_names(self):
        courses_names = [
            re.search(r'\r(.+)\r|$', field).group()
            for field in self.df.columns
        ]
        return [
            re.sub(
                r'(?<=[A-Z])\r(?=[A-Z])|$',
                '',
                field)
            .replace('\r', ' ')
            .strip()
            for field in courses_names
        ]

    """
    Rename UEs labels in header
    """
    def rename_header(self, ue_id=False):
        names = self.df.columns
        modules = self.courses_ids if ue_id else self.courses_names
        self.df.columns = list(names)[:3] + modules[3:]

    def get_dataframe(self):
        return self.df

def parse_args():
    parser = argparse.ArgumentParser(
        prog="APOGEE PDF to CSV",
        description="Extracts APOGEE data from PDF"
    )
    parser.add_argument(
        "filename",
        help="PDF file",
        type=Path
    )
    parser.add_argument(
        "-o", "--output",
        help="output filename",
        type=Path
    )
    parser.add_argument(
        "-i", "--id",
        help="labels modules with their ID",
        action='store_true',
        default=False,
    )

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    pv = RefactoredPDF(args.filename)
    df = pv.get_dataframe()
    
    if args.id:
        pv.rename_header(ue_id=True)
    
    df.to_csv(args.output, index=False)
