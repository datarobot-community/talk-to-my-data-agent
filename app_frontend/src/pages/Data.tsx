import React, { useState } from 'react';
import { Separator } from '@radix-ui/react-separator';
import { useGeneratedDictionaries } from '@/api/dictionaries/hooks';

import {
    DatasetCardDescriptionPanel,
    DataViewTabs,
    SearchControl,
    ClearDatasetsButton,
} from '@/components/data';
import { ValueOf } from '@/state/types';
import { DATA_TABS } from '@/state/constants';
import { Loading } from '@/components/ui-custom/loading';

export const Data: React.FC = () => {
    const { data, status } = useGeneratedDictionaries();
    const [viewMode, setViewMode] = useState<ValueOf<typeof DATA_TABS>>(DATA_TABS.DESCRIPTION);

    return (
        <div className="p-6">
            <h2 className="text-xl">
                <strong>Data</strong>
            </h2>
            <div className="flex justify-between gap-2">
                <div className="flex gap-2 items-center">
                    <div className="text-sm">View</div>
                    <DataViewTabs
                        defaultValue={viewMode}
                        onChange={value => setViewMode(value as ValueOf<typeof DATA_TABS>)}
                    />
                    <SearchControl />
                </div>
                <div className="flex items-center">
                    <ClearDatasetsButton />
                </div>
            </div>
            <Separator className="my-4 border-t" />
            {status === 'pending' ? (
                <div className="flex items-center justify-center h-[calc(100vh-200px)]">
                    <Loading />
                </div>
            ) : (
                <div className="flex flex-col gap-4">
                    {data?.map(dictionary => (
                        <DatasetCardDescriptionPanel
                            key={dictionary.name}
                            isProcessing={dictionary.in_progress || false}
                            dictionary={dictionary}
                            viewMode={viewMode}
                        />
                    ))}
                </div>
            )}
        </div>
    );
};
