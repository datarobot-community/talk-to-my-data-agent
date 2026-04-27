import { DictionaryTable } from './types';
import { Loader2, TriangleAlert } from 'lucide-react';

export const getDictionariesMenu = (data: DictionaryTable[]) =>
  data?.map(dictionary => ({
    key: dictionary.name,
    name: dictionary.name,
    endIcon: dictionary.error ? (
      <TriangleAlert className="mr-2 size-4 text-destructive" />
    ) : dictionary.in_progress ? (
      <Loader2 className="mr-2 size-4 animate-spin" />
    ) : undefined,
  }));
