import apiClient from '../apiClient';

type DatabaseTables = Array<string>;

export const getDatabaseTables = async ({
  signal,
}: {
  signal?: AbortSignal;
}): Promise<DatabaseTables> => {
  const { data } = await apiClient.get<DatabaseTables>(`/v1/database/tables`, {
    signal,
    // A failed table discovery is surfaced to the user as an outage, so don't
    // let the client hammer a down database — no axios-level retries here.
    'axios-retry': { retries: 0 },
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
