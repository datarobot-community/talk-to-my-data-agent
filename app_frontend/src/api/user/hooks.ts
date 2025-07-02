import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { getDataRobotInfo, updateApiToken } from './api-requests';
import { dataRobotInfoKey } from './keys';
import { useEffect } from 'react';
import { datasetKeys } from '../datasets/keys';

export const useDataRobotInfo = () => {
    const query = useQuery({
        queryKey: dataRobotInfoKey,
        queryFn: getDataRobotInfo,
    });

    useEffect(() => {
        try {
            getDataRobotInfo();
        } catch (error) {
            console.error('Error in DataRobot info effect:', error);
        }
    }, []);

    return query;
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
