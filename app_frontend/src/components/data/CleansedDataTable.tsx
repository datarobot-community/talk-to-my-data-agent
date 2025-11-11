import React, { useEffect, useMemo } from 'react';
import { useInView } from 'react-intersection-observer';
import { useInfiniteCleansedDataset } from '@/api/cleansed-datasets/hooks';
import { useInfiniteDatasetById } from '@/api/datasets/hooks';
import { TableHeader, TableRow, TableHead, TableBody, TableCell } from '@/components/ui/table';
import loader from '@/assets/loader.svg';
import { Loading } from '@/components/ui-custom/loading';
import { useTranslation } from '@/i18n';
import { HighlightText } from '@/components/ui-custom/highlight-text';

interface CleansedDataTableProps {
  datasetName?: string;
  datasetId?: string;
  rowsPerPage?: number;
  searchText?: string;
  maxHeight?: string;
  className?: string;
}

export const CleansedDataTable: React.FC<CleansedDataTableProps> = ({
  datasetName,
  datasetId,
  rowsPerPage = 50,
  searchText,
  maxHeight = 'max-h-[600px]',
  className = '',
}) => {
  const { ref, inView } = useInView();
  const { t } = useTranslation();

  const cleansedDataQuery = useInfiniteCleansedDataset(
    datasetName || '',
    rowsPerPage,
    searchText,
    !datasetId && !!datasetName
  );
  const datasetByIdQuery = useInfiniteDatasetById(datasetId, {
    pageSize: rowsPerPage,
    enabled: !!datasetId,
  });

  const activeQuery = datasetId ? datasetByIdQuery : cleansedDataQuery;
  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, status, isError, error } =
    activeQuery;

  useEffect(() => {
    if (inView && hasNextPage && !isFetchingNextPage) {
      fetchNextPage();
    }
  }, [inView, fetchNextPage, hasNextPage, isFetchingNextPage]);

  // Create a flat array of all rows from all pages
  const allRows = useMemo(() => {
    if (!data) return [];
    return data.pages.flatMap(page => page.dataset.data_records);
  }, [data]);

  // Get column headers from the first page if available
  const columns = useMemo(() => {
    if (!data?.pages || data.pages.length === 0) return [];

    const firstPage = data.pages[0];
    const firstRecord = firstPage.dataset?.data_records?.[0];

    return firstRecord ? Object.keys(firstRecord) : [];
  }, [data]);

  if (status === 'pending') {
    return (
      <div className="h-96">
        <Loading />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex flex-col flex-1 items-center justify-center h-96">
        {t('Error loading data')}: {String(error)}
      </div>
    );
  }

  if (allRows.length === 0) {
    return (
      <div className="flex flex-col flex-1 items-center justify-center h-96">
        {t('No data available for this dataset.')}
      </div>
    );
  }

  return (
    <div className={`w-0 min-w-full ${className}`}>
      <div className={`overflow-auto ${maxHeight}`}>
        <table className="w-full caption-bottom text-sm">
          <TableHeader className="bg-background sticky top-0 z-10">
            <TableRow>
              {columns.map(column => (
                <TableHead key={column} className="whitespace-nowrap">
                  <HighlightText text={column} searchText={searchText || ''} />
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {allRows.map((row, index) => (
              <TableRow key={index}>
                {columns.map(column => (
                  <TableCell key={column} className="whitespace-nowrap">
                    {row[column] !== null && row[column] !== undefined ? String(row[column]) : ''}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </table>

        {/* Loading indicator - inside scrollable area */}
        <div ref={ref} className="w-full text-center p-4">
          {isFetchingNextPage ? (
            <div className="flex justify-center items-center">
              <img src={loader} alt={t('processing')} className="mr-2 w-4 h-4 animate-spin" />
              <span className="ml-2">{t('Loading more...')}</span>
            </div>
          ) : (
            <div className="h-4" />
          )}
        </div>
      </div>
    </div>
  );
};
