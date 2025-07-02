import { useInfiniteQuery, useQuery } from '@tanstack/react-query';
import { cleansedDatasetKeys, datasetMetadataKeys } from './keys';
import { getCleansedDataset, getDatasetMetadata } from './api-requests';

export const useInfiniteCleansedDataset = (name: string, limit = 100) => {
    return useInfiniteQuery({
        queryKey: cleansedDatasetKeys.detail(name),
        initialPageParam: 0,
        queryFn: ({ pageParam = 0, signal }) =>
            getCleansedDataset({
                name,
                skip: pageParam,
                limit,
                signal,
            }),
        getNextPageParam: (lastPage, allPages) => {
            const totalFetched = allPages.length * limit;
            // If we received fewer rows than the limit, we've reached the end
            if (lastPage.dataset.data_records.length < limit) return undefined;
            return totalFetched;
        },
        // Keep data for 5 minutes before refetching
        staleTime: 5 * 60 * 1000,
    });
};

export const useDatasetMetadata = (name: string) => {
    return useQuery({
        queryKey: datasetMetadataKeys.detail(name),
        queryFn: ({ signal }) => getDatasetMetadata({ name, signal }),
        // Keep data for 5 minutes before refetching
        staleTime: 5 * 60 * 1000,
        // Don't refetch on window focus for metadata (doesn't change frequently)
        refetchOnWindowFocus: false,
    });
};

export const useMultipleDatasetMetadata = (names: string[]) => {
    return useQuery({
        queryKey: datasetMetadataKeys.list(names),
        queryFn: async ({ signal }) => {
            const results = [];
            for (const name of names) {
                const metadata = await getDatasetMetadata({ name, signal });
                results.push({ name, metadata });
            }
            return results;
        },
        // Keep data for 5 minutes before refetching
        staleTime: 5 * 60 * 1000,
        // Don't refetch on window focus for metadata (doesn't change frequently)
        refetchOnWindowFocus: false,
    });
};
