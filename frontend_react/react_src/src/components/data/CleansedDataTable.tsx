import React, { useEffect, useMemo } from "react";
import { useInView } from "react-intersection-observer";
import { useInfiniteCleansedDataset } from "@/api-state/cleansed-datasets/hooks";
import {
  Table,
  TableHeader,
  TableRow,
  TableHead,
  TableBody,
  TableCell,
} from "@/components/ui/table";
import loader from "@/assets/loader.svg";
import { Loading } from "@/components/ui-custom/loading";
import { useTranslation } from "react-i18next";

interface CleansedDataTableProps {
  datasetName: string;
  rowsPerPage?: number;
}

export const CleansedDataTable: React.FC<CleansedDataTableProps> = ({
  datasetName,
  rowsPerPage = 50,
}) => {
  const { ref, inView } = useInView();
  const { t } = useTranslation();

  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    status,
    isError,
    error,
  } = useInfiniteCleansedDataset(datasetName, rowsPerPage);

  useEffect(() => {
    if (inView && hasNextPage && !isFetchingNextPage) {
      fetchNextPage();
    }
  }, [inView, fetchNextPage, hasNextPage, isFetchingNextPage]);

  // Create a flat array of all rows from all pages
  const allRows = useMemo(() => {
    if (!data) return [];
    return data.pages.flatMap((page) => page.dataset.data_records);
  }, [data]);

  // Get column headers from the first page if available
  const columns = useMemo(() => {
    if (!data || !data.pages[0]?.dataset?.data_records[0]) return [];
    return Object.keys(data.pages[0].dataset.data_records[0]);
  }, [data]);

  if (status === "pending") {
    return (
      <div className="h-96">
        <Loading />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex flex-col flex-1 items-center justify-center h-96">
        {t("error_loading_data")} {String(error)}
      </div>
    );
  }

  if (allRows.length === 0) {
    return (
      <div className="flex flex-col flex-1 items-center justify-center h-96">
        {t("no_data_available")}
      </div>
    );
  }

  return (
    <div className="w-full overflow-auto">
      <Table>
        <TableHeader className="bg-background">
          <TableRow>
            {columns.map((column) => (
              <TableHead key={column}>{column}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {allRows.map((row, index) => (
            <TableRow key={index}>
              {columns.map((column) => (
                <TableCell key={column}>
                  {row[column] !== null && row[column] !== undefined
                    ? String(row[column])
                    : ""}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>

      {/* Loading indicator */}
      <div ref={ref} className="w-full text-center p-4">
        {isFetchingNextPage ? (
          <div className="flex justify-center items-center">
            <img
              src={loader}
              alt={t("processing")}
              className="mr-2 w-4 h-4 animate-spin"
            />
            <span className="ml-2">{t("loading")}</span>
          </div>
        ) : hasNextPage ? (
          <div className="h-10" />
        ) : (
          <div className="text-muted-foreground">
            {allRows.length > 0 ? t("end_of_data") : t("no_data")}
          </div>
        )}
      </div>
    </div>
  );
};
