import * as React from 'react';
import * as LabelPrimitive from '@radix-ui/react-label';
import { cn } from '@/lib/utils';

function Label({ className, htmlFor, ...props }: React.ComponentProps<typeof LabelPrimitive.Root>) {
    const isForEnabledRadio = React.useMemo(() => {
        if (typeof document !== 'undefined' && htmlFor) {
            const element = document.getElementById(htmlFor);
            if (element && element instanceof HTMLInputElement && element.type === 'radio') {
                return !element.disabled;
            }
        }
        return false;
    }, [htmlFor]);

    return (
        <LabelPrimitive.Root
            data-slot="label"
            htmlFor={htmlFor}
            className={cn(
                'text-sm leading-none font-medium select-none group-data-[disabled=true]:pointer-events-none group-data-[disabled=true]:opacity-50 peer-disabled:cursor-not-allowed peer-disabled:opacity-50',
                !isForEnabledRadio && 'cursor-pointer',
                className
            )}
            {...props}
        />
    );
}

export { Label };
