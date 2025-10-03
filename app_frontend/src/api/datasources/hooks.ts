import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { dataSourceKeys } from './keys';
import {
  externalDataSourceName,
  ExternalDataStore,
  listAvailableExternalDataStores,
  selectSourcesForDataStore,
} from './api-requests';
import { dictionaryKeys, DictionaryTable } from '../dictionaries';

export const useListAvailableDataStores = () => {
  const queryResult = useQuery({
    queryKey: dataSourceKeys.available,
    queryFn: listAvailableExternalDataStores,
    refetchInterval: 5 * 60 * 1000,
  });
  return queryResult;
};

export const useSelectDataSourcesMutation = ({
  onSuccess,
  onError,
}: {
  onSuccess: (data: unknown) => void;
  onError: (error: Error) => void;
}) => {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: async ({
      selectedDataStore,
      selectedDataSourceNames,
    }: {
      selectedDataStore: ExternalDataStore;
      selectedDataSourceNames: string[];
    }) => {
      const selectedDataSourceNamesSet = new Set(selectedDataSourceNames);
      const selectedDataSources = selectedDataStore.defined_data_sources.filter(d =>
        selectedDataSourceNamesSet.has(externalDataSourceName(d))
      );
      await selectSourcesForDataStore(selectedDataStore.id, selectedDataSources);
      return { selectedDataStore, selectedDataSources };
    },
    onMutate: async ({ selectedDataSourceNames }) => {
      const previousDictionaries =
        queryClient.getQueryData<DictionaryTable[]>(dictionaryKeys.all) || [];

      const previousDictionaryNames = new Set(previousDictionaries.map(d => d.name));

      const placeholderDictionaries: DictionaryTable[] = selectedDataSourceNames
        .filter(name => !previousDictionaryNames.has(name))
        .map(name => ({
          name: name,
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
      onSuccess(data);
      await queryClient.invalidateQueries({ queryKey: dictionaryKeys.all });
    },
    onError: (error, _data, context) => {
      onError(error);
      if (context) {
        queryClient.setQueryData<DictionaryTable[]>(
          dictionaryKeys.all,
          context.previousDictionaries
        );
      }
    },
    onSettled: async () => {
      await queryClient.refetchQueries({ queryKey: dataSourceKeys.registered });
    },
  });

  return mutation;
};
