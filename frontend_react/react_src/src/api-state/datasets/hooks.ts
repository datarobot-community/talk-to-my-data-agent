import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { datasetKeys } from "./keys";
import { getDatasets, uploadDataset, deleteAllDatasets } from "./api-requests";
import { useState } from "react";
import { dictionaryKeys } from "../dictionaries/keys";
import { DictionaryTable } from "../dictionaries/types";
import { AxiosError } from "axios";

export interface FileUploadResponse {
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
  filenames?: string;
}

export const useFetchAllDatasets = ({ limit = 100 } = {}) => {
  const queryResult = useQuery({
    queryKey: datasetKeys.all,
    queryFn: ({ signal }) => getDatasets({ signal, limit }),
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
    }: {
      files: File[];
      catalogIds: string[];
    }) => {
      const response = await uploadDataset({
        files,
        catalogIds,
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const prg = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setProgress(prg);
          }
        },
      });
      if (Array.isArray(response)) {
        const datasetsWithError = response.filter(
          (file: FileUploadResponse) => file.error
        );
        if (datasetsWithError.length > 0) {
          const filenames = datasetsWithError
            .map((file) => file.filename || file.dataset_name)
            .filter(Boolean);
          let filenamesStr = "";
          if (filenames.length === 1) {
            filenamesStr = filenames[0] || "";
          } else if (filenames.length === 2) {
            filenamesStr = `${filenames[0]} and ${filenames[1]}`;
          } else if (filenames.length > 2) {
            filenamesStr = `${filenames.slice(0, -1).join(", ")} and ${filenames[filenames.length - 1]}`;
          }
          const error = new Error("upload_file_error");
          (error as UploadError).responseData = response;
          (error as UploadError).filenames = filenamesStr;
          throw error;
        }
        return response;
      }

      const error = new Error("upload_network_error");
      throw error;
    },
    onMutate: async ({ files }) => {
      const previousDictionaries =
        queryClient.getQueryData<DictionaryTable[]>(dictionaryKeys.all) || [];

      const placeholderDictionaries: DictionaryTable[] = files.map((file) => ({
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
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: dictionaryKeys.all });
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
      } else if (
        "isAxiosError" in error &&
        error.isAxiosError &&
        (error as AxiosError).response
      ) {
        const axiosError = error as AxiosError;
        uploadError.response = {
          data: axiosError.response?.data,
        };
      }

      onError(uploadError);
    },
    onSettled: () =>
      queryClient.invalidateQueries({ queryKey: datasetKeys.all }),
  });

  return { ...mutation, progress };
};

export const useDeleteAllDatasets = () => {
  const queryClient = useQueryClient();
  const mutation = useMutation({
    mutationFn: () => deleteAllDatasets(),
    onMutate: async () => {
      await queryClient.cancelQueries({ queryKey: dictionaryKeys.all });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: dictionaryKeys.all });
    },
  });
  return mutation;
};
