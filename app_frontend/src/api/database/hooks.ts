import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { databaseKeys } from './keys';
import { getDatabaseTables, loadFromDatabase } from './api-requests';
import { dictionaryKeys } from '../dictionaries/keys';
import { DictionaryTable } from '../dictionaries/types';

export const useGetDatabaseTables = (enabled = true) => {
  const queryResult = useQuery({
    queryKey: databaseKeys.all,
    queryFn: ({ signal }) => getDatabaseTables({ signal }),
    enabled,
    // Surface an outage on the first attempt rather than retrying a down database.
    retry: false,
    // An errored query is stale, so window refocus would otherwise re-ping a
    // down database on every focus while the modal is open.
    refetchOnWindowFocus: false,
  });

  return queryResult;
};

export const useLoadFromDatabaseMutation = ({
  onSuccess,
  onError,
}: {
  onSuccess: (data: unknown) => void;
  onError: (error: Error) => void;
}) => {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: ({ tableNames }: { tableNames: string[] }) =>
      loadFromDatabase({
        tableNames,
      }),
    onMutate: async () => {
      const previousDictionaries =
        queryClient.getQueryData<DictionaryTable[]>(dictionaryKeys.all) || [];
      return { previousDictionaries };
    },
    onSuccess: data => {
      queryClient.invalidateQueries({ queryKey: dictionaryKeys.all });
      onSuccess(data);
    },
    onError: (error, _, context) => {
      if (context?.previousDictionaries) {
        queryClient.setQueryData<DictionaryTable[]>(
          dictionaryKeys.all,
          context.previousDictionaries
        );
      }
      onError(error);
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: databaseKeys.all }),
  });

  return mutation;
};
