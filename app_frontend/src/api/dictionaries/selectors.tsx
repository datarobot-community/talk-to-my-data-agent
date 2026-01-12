import { DictionaryTable } from './types';
import { Loader2 } from 'lucide-react';

export const getDictionariesMenu = (data: DictionaryTable[]) =>
  data?.map(dictionary => ({
    key: dictionary.name,
    name: dictionary.name,
    endIcon: dictionary.in_progress ? <Loader2 className="mr-2 w-4 h-4 animate-spin" /> : undefined,
  }));
