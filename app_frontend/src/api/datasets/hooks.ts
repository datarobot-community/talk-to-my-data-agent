import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { datasetKeys } from './keys';
import {
  getDatasets,
  uploadDataset,
  deleteAllDatasets,
  getSupportedDataSourceTypes,
} from './api-requests';
import { useState } from 'react';
import { dictionaryKeys } from '../dictionaries/keys';
import { DictionaryTable } from '../dictionaries/types';
import { AxiosError } from 'axios';

interface FileUploadResponse {
  filename?: string;
  content_type?: string;
  size?: number;
  dataset_name?: string;
  error?: string;
}

export interface UploadError extends Error {
  responseData?: FileUploadResponse[];
  response?: {
    data: unknown;
  };
  isAxiosError?: boolean;
}

export const useFetchDatasets = ({ limit = 100 } = {}) => {
  const queryResult = useQuery({
    queryKey: datasetKeys.list(limit),
    queryFn: async ({ signal }) => {
      const [local, remote] = await Promise.all([
        getDatasets({ signal, limit, remote: false }),
        getDatasets({ signal, limit, remote: true }),
      ]);
      return { local: local, remote: remote };
    },
    refetchInterval: 5 * 60 * 1000,
  });

  return queryResult;
};

export const useGetSupportedDataSourceTypes = () => {
  const queryResult = useQuery({
    queryKey: datasetKeys.supportedDataSourceTypes,
    queryFn: getSupportedDataSourceTypes,
  });

  return queryResult;
};

export const useFileUploadMutation = ({
  onSuccess,
  onError,
}: {
  onSuccess: (data: unknown) => void;
  onError: (error: UploadError | AxiosError) => void;
}) => {
  const [progress, setProgress] = useState(0);
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: async ({
      files,
      catalogIds,
      dataSource,
    }: {
      files: File[];
      catalogIds: string[];
      dataSource: string;
    }) => {
      const response = await uploadDataset({
        files,
        catalogIds,
        dataSource,
        onUploadProgress: progressEvent => {
          if (progressEvent.total) {
            const prg = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setProgress(prg);
          }
        },
      });
      if (Array.isArray(response)) {
        const datasetsWithError = response.filter((file: FileUploadResponse) => file.error);
        if (datasetsWithError.length > 0) {
          let message = '';
          for (const datasetWithError of datasetsWithError) {
            message = `Error uploading ${
              datasetWithError.filename || datasetWithError.dataset_name
            }: ${datasetWithError.error} \n\n${message}`;
          }

          const error = new Error(message) as UploadError;
          error.responseData = response;
          throw error;
        }

        return response;
      }
    },
    onMutate: async ({ files }) => {
      const previousDictionaries =
        queryClient.getQueryData<DictionaryTable[]>(dictionaryKeys.all) || [];

      const placeholderDictionaries: DictionaryTable[] = files.map(file => ({
        name: file.name,
        in_progress: true,
        column_descriptions: [],
      }));

      queryClient.setQueryData<DictionaryTable[]>(dictionaryKeys.all, [
        ...previousDictionaries,
        ...placeholderDictionaries,
      ]);

      return { previousDictionaries };
    },
    onSuccess: async data => {
      await queryClient.invalidateQueries({ queryKey: dictionaryKeys.all });
      onSuccess(data);
    },
    onError: (error: UploadError | AxiosError, _, context) => {
      if (context?.previousDictionaries) {
        queryClient.setQueryData<DictionaryTable[]>(
          dictionaryKeys.all,
          context.previousDictionaries
        );
      }

      const uploadError = error as UploadError;

      if (uploadError.responseData) {
        uploadError.response = { data: uploadError.responseData };
      } else if ('isAxiosError' in error && error.isAxiosError && (error as AxiosError).response) {
        const axiosError = error as AxiosError;
        uploadError.response = {
          data: axiosError.response?.data,
        };
      }

      onError(uploadError);
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: datasetKeys.all }),
  });

  return { ...mutation, progress };
};

export const useDeleteAllDatasets = ({ onSuccess }: { onSuccess?: () => void }) => {
  const queryClient = useQueryClient();
  const mutation = useMutation({
    mutationFn: () => deleteAllDatasets(),
    onMutate: async () => {
      await queryClient.cancelQueries({ queryKey: dictionaryKeys.all });
    },
    onSuccess: () => {
      queryClient.setQueryData<DictionaryTable[]>(dictionaryKeys.all, []);
      queryClient.invalidateQueries({ queryKey: dictionaryKeys.all });
      onSuccess?.();
    },
  });
  return mutation;
};
