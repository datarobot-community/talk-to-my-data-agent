import { toast } from 'sonner';
import i18n from '@/i18n';
import apiClient from '../apiClient';
import { DictionaryRow, DictionaryTable } from './types';

export const getGeneratedDictionaries = async ({
  signal,
}: {
  signal?: AbortSignal;
}): Promise<Array<DictionaryTable>> => {
  const { data } = await apiClient.get<Array<DictionaryTable>>(`/v1/dictionaries`, {
    signal,
  });

  return data;
};

export const deleteGeneratedDictionary = async ({
  name,
  signal,
}: {
  name: string;
  signal?: AbortSignal;
}): Promise<void> => {
  const encodedName = encodeURIComponent(name);
  await apiClient.delete(`/v1/dictionaries/${encodedName}`, { signal });

  return;
};

export const updateDictionaryCell = async ({
  name,
  rowIndex,
  field,
  value,
  signal,
}: {
  name: string;
  rowIndex: number;
  field: keyof DictionaryRow;
  value: string;
  signal?: AbortSignal;
}): Promise<DictionaryTable> => {
  const encodedName = encodeURIComponent(name);
  const { data } = await apiClient.patch<DictionaryTable>(
    `/v1/dictionaries/${encodedName}/cells`,
    {
      rowIndex,
      field,
      value,
    },
    { signal }
  );

  return data;
};

export const downloadDictionary = async ({
  name,
  signal,
  includeBom,
}: {
  name: string;
  signal?: AbortSignal;
  includeBom?: boolean;
}): Promise<void> => {
  const encodedName = encodeURIComponent(name);

  try {
    // Use fetch directly instead of apiClient to get raw response with blob
    const response = await fetch(
      `${apiClient.defaults.baseURL}/v1/dictionaries/${encodedName}/download${
        includeBom ? '?bom=true' : ''
      }`,
      {
        method: 'GET',
        headers: {
          Accept: 'application/json, text/csv',
          'Content-Type': 'application/json',
        },
        credentials: 'include', // Include cookies in the request
        signal,
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to download dictionary: ${response.statusText}`);
    }

    const filename = i18n.t('{{name}}_dictionary.csv', { name });
    // Convert response to blob
    const blob = await response.blob();

    // Create download link and trigger download
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();

    // Clean up
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  } catch (error) {
    console.error('Error downloading dictionary:', error);
    toast.error(i18n.t('Error downloading dictionary'));
    throw error;
  }
};
