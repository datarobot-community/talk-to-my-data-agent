import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { getDataRobotInfo, updateApiToken } from './api-requests';
import { dataRobotInfoKey } from './keys';
import { datasetKeys } from '../datasets/keys';

export const useDataRobotInfo = () => {
  return useQuery({
    queryKey: dataRobotInfoKey,
    queryFn: getDataRobotInfo,
  });
};

export const useUpdateApiToken = () => {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: updateApiToken,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: dataRobotInfoKey });
      queryClient.invalidateQueries({ queryKey: datasetKeys.all });
    },
  });

  return mutation;
};
