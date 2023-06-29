#!/bin/python3

import re
import argparse
from tabula import read_pdf
from functools import reduce, cached_property
from decimal import Decimal
from copy import deepcopy
from pathlib import Path
# from IPython.display import display
import pandas

def list_diff(l1: list, l2: list) -> list[int]:
    return [i for i, (e1, e2) in enumerate(zip(l1, l2)) if e1 != e2]

class RefactoredPDF:
    def __init__(self, filename: str):
        self.filename = filename
        self.dfs = read_pdf(
            filename,
            multiple_tables=True,
            lattice=True,
            pages="all",
            encoding='ISO-8859-1'
        )
        self.split_count
        self.check_lines_count()
        self.refactor_columns_names()
        self.concat_columns()
        self.drop_stats_line()
        self.split_resultats()
        self.normalize()
        self.courses_ids
        self.courses_names
        self.rename_header()

    @cached_property
    def sheet_count(self) -> int:
        return len(self.dfs)

    @cached_property
    def split_count(self) -> int:
        cols_len = [len(df.columns) for df in self.dfs]
        split_count = cols_len.index(min(cols_len)) + 1
        diff = list_diff((self.sheet_count // split_count) * cols_len[:split_count], cols_len)
        
        if diff:
            raise ValueError(
                f"Thoses pages doesn't have an inconsistant number of columns, "
                f"compared to the first group : {diff}")

        return split_count

    def check_lines_count(self) -> bool:
        lines_len = [len(d) for d in self.dfs]
        for i in range(self.sheet_count, self.split_count):
            for length in lines_len[i:i+self.split_count]:
                if length != lines_len[0]:
                    raise ValueError(
                        f"Page n°{i} have an inconsistant number of lines : "
                        f"{length}, expected {lines_len[0]}")
        return True

    def refactor_columns_names(self):
        for df in self.dfs:
            cols = list(df.columns)
            cols.insert(1, '')
            cols.pop()
            cols[0] = "NumEtudiant"
            df.columns = cols
            df.drop('', axis=1, inplace=True)
    
    def concat_columns(self):
        merged = []
        # print(len(self.dfs))
        # Merge columns that were splitted accros sheets
        for first_page in range(0, self.sheet_count, self.split_count):
            splitted_pages = [self.dfs[i] for i in range(first_page, first_page+self.split_count, 1)]
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
        self.dfs = pandas.concat(merged, axis=0)
        self.dfs.reset_index(drop=True, inplace=True)

    def drop_stats_line(self):
        self.dfs.drop(self.dfs.tail(1).index, inplace=True)

    def split_resultats(self):
        resultats = self.dfs[self.dfs.columns[1]]
        
        # Split 'results' column that contains 'admission' status 
        resultats = resultats.str.split('\r', expand=True).iloc[:, :2]
        
        # Force two columns if only one was found
        indexes = (0,1) if len(resultats.columns) == 2 else (1,0)
        resultats = resultats.reindex(labels=indexes, axis=1, fill_value=float('NaN'))
        
        resultats.columns = ["Résultat", "Admission"]
        self.dfs = pandas.concat([self.dfs.iloc[:, 0:1], resultats, self.dfs.iloc[:, 2:]], axis=1)

    def normalize(self):
        def extract_number(cell: str):
            if match := re.search('(\\d*\\.?\\d+)', str(cell)):
                return Decimal(match.group())
            return Decimal("NaN")

        left  = self.dfs.iloc[:, 0:2].applymap(extract_number)
        right = self.dfs.iloc[:, 3: ].applymap(extract_number)

        self.dfs = self.dfs.assign(**left).assign(**right)

    """
    Extract all UE codes from header
    """
    @cached_property
    def courses_ids(self):
        return [re.search("^([A-Z]|\\d){7}|$", field).group() for field in self.dfs.columns]

    """
    Extract all UE names from header
    """
    @cached_property
    def courses_names(self):
        courses_names = [re.search(r'\r(.+)\r|$', field).group() for field in self.dfs.columns]
        return [re.sub(r'(?<=[A-Z])\r(?=[A-Z])|$', '', field).replace('\r', ' ').strip() for field in courses_names]

    """
    Rename UEs labels in header
    """
    def rename_header(self, ue_id=False):
        names = self.dfs.columns
        modules = self.courses_ids if ue_id else self.courses_names
        self.dfs.columns = list(names)[:3] + modules[3:]

    def get_dataframe(self):
        return self.dfs

def parse_args():
    parser = argparse.ArgumentParser(
        prog="APOGEE Extractor",
        description="Extracts APOGEE data from PDF"
    )
    parser.add_argument(
        "-f", "--file",
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
    pv = RefactoredPDF(args.file)
    df = pv.get_dataframe()
    
    if args.id:
        pv.rename_header(ue_id=True)
    
    df.to_csv(args.output, index=False)
