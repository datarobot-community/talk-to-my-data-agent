import React from 'react';
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table';

interface AnalystDatasetTableProps {
    records?: Record<string, unknown>[];
}

export const AnalystDatasetTable: React.FC<AnalystDatasetTableProps> = ({ records }) => {
    const headerRow = records?.length ? Object.keys(records[0]) : [];
    return (
        <Table>
            <TableHeader className="bg-background">
                <TableRow>
                    {headerRow.map(h => (
                        <TableHead key={h}>{h}</TableHead>
                    ))}
                </TableRow>
            </TableHeader>
            <TableBody>
                {records?.map((record, index) => (
                    <TableRow key={index}>
                        {Object.keys(record).map(k => (
                            <TableCell key={`${index}_${k}`}>{String(record[k])}</TableCell>
                        ))}
                    </TableRow>
                ))}
            </TableBody>
        </Table>
    );
};
