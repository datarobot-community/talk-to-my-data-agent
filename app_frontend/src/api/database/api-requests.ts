import apiClient from '../apiClient';

type DatabaseTables = Array<string>;

export const getDatabaseTables = async ({
  signal,
}: {
  signal?: AbortSignal;
}): Promise<DatabaseTables> => {
  const { data } = await apiClient.get<DatabaseTables>(`/v1/database/tables`, {
    signal,
  });
  return data;
};

export const loadFromDatabase = async ({
  tableNames,
  signal,
}: {
  tableNames: string[];
  signal?: AbortSignal;
}): Promise<string[]> => {
  const { data } = await apiClient.post<string[]>(
    '/v1/database/select',
    { table_names: tableNames },
    {
      signal,
    }
  );
  return data;
};
