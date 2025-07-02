import { useQuery, useQueryClient, useMutation } from '@tanstack/react-query';
import { dictionaryKeys } from './keys';
import {
    getGeneratedDictionaries,
    deleteGeneratedDictionary,
    updateDictionaryCell,
    downloadDictionary,
} from './api-requests';
import { DictionaryRow, DictionaryTable } from './types';

export const useGeneratedDictionaries = () => {
    const queryResult = useQuery<DictionaryTable[]>({
        queryKey: dictionaryKeys.all,
        queryFn: ({ signal }) => getGeneratedDictionaries({ signal }),
        refetchInterval: query =>
            !query || query.state?.data?.some(d => d.in_progress) ? 5000 : false,
    });

    return queryResult;
};

export const useDeleteGeneratedDictionary = () => {
    const queryClient = useQueryClient();
    const mutation = useMutation({
        mutationFn: ({ name }: { name: string }) => deleteGeneratedDictionary({ name }),
        onMutate: async () => {
            await queryClient.cancelQueries({ queryKey: dictionaryKeys.all });
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: dictionaryKeys.all });
        },
    });

    return mutation;
};

export const useUpdateDictionaryCell = () => {
    const queryClient = useQueryClient();
    const mutation = useMutation<
        DictionaryTable,
        Error,
        {
            name: string;
            rowIndex: number;
            field: keyof DictionaryRow;
            value: string;
        },
        {
            previousDictionaries: DictionaryTable[];
            originalValue: string;
        }
    >({
        mutationFn: ({ name, rowIndex, field, value }) =>
            updateDictionaryCell({ name, rowIndex, field, value }),
        onMutate: async ({ name, rowIndex, field, value }) => {
            await queryClient.cancelQueries({ queryKey: dictionaryKeys.all });

            const previousDictionaries =
                queryClient.getQueryData<DictionaryTable[]>(dictionaryKeys.all) || [];

            // Store the original value for error case
            let originalValue = '';
            const dictionary = previousDictionaries.find(d => d.name === name);
            if (dictionary?.column_descriptions?.[rowIndex]) {
                originalValue = dictionary.column_descriptions[rowIndex][field];
            }

            queryClient.setQueryData<DictionaryTable[]>(dictionaryKeys.all, oldData => {
                if (!oldData) return previousDictionaries;

                return oldData.map(dictionary => {
                    if (dictionary.name !== name) return dictionary;

                    const updatedDictionary = { ...dictionary };

                    if (
                        updatedDictionary.column_descriptions &&
                        updatedDictionary.column_descriptions[rowIndex]
                    ) {
                        updatedDictionary.column_descriptions = [
                            ...updatedDictionary.column_descriptions,
                        ];
                        updatedDictionary.column_descriptions[rowIndex] = {
                            ...updatedDictionary.column_descriptions[rowIndex],
                            [field]: value,
                        };
                    }

                    return updatedDictionary;
                });
            });

            return {
                previousDictionaries,
                originalValue,
            };
        },
        onError: (_, __, context) => {
            if (context?.previousDictionaries) {
                queryClient.setQueryData(dictionaryKeys.all, context.previousDictionaries);
            }
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: dictionaryKeys.all });
        },
    });

    return mutation;
};

export const useDownloadDictionary = () => {
    return useMutation({
        mutationFn: downloadDictionary,
    });
};
