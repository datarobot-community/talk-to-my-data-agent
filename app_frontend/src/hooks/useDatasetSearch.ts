import { useState, useMemo } from 'react';
import { DictionaryTable as DT, DictionaryRow } from '@/api/dictionaries/types';

const COLUMNS_TO_SEARCH = [
  'column',
  'data_type',
  'description',
] as const satisfies (keyof DictionaryRow)[];

export const useDatasetDictionarySearch = (dictionary: DT) => {
  const [searchText, setSearchText] = useState('');

  const filteredDictionary = useMemo(() => {
    if (!searchText.trim()) {
      return dictionary;
    }

    const lowercaseSearch = searchText.toLowerCase();
    const filteredColumns = dictionary.column_descriptions?.filter(column => {
      return COLUMNS_TO_SEARCH.some(field => {
        const value = column[field];
        return value && value.toLowerCase().includes(lowercaseSearch);
      });
    });

    return {
      ...dictionary,
      column_descriptions: filteredColumns,
    };
  }, [dictionary, searchText]);

  const getOriginalRowIndex = (filteredRowIndex: number): number => {
    if (!searchText.trim() || !filteredDictionary.column_descriptions) {
      return filteredRowIndex;
    }

    const filteredColumn = filteredDictionary.column_descriptions[filteredRowIndex];
    if (!filteredColumn) {
      return filteredRowIndex;
    }

    return (
      dictionary.column_descriptions?.findIndex(col => col.column === filteredColumn.column) ??
      filteredRowIndex
    );
  };

  return {
    searchText,
    setSearchText,
    filteredDictionary,
    getOriginalRowIndex,
  };
};
