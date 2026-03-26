'use client';

import * as React from 'react';
import * as SwitchPrimitives from '@radix-ui/react-switch';

import { cn } from '@/lib/utils';

const Switch = React.forwardRef<
  React.ElementRef<typeof SwitchPrimitives.Root>,
  React.ComponentPropsWithoutRef<typeof SwitchPrimitives.Root>
>(({ className, ...props }, ref) => (
  <SwitchPrimitives.Root
    className={cn(
      'peer inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full border-1 border-transparent shadow-sm transition-colors hover:not-disabled:border-transparent focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-70 data-[state=checked]:bg-accent hover:data-[state=checked]:bg-accent/80 disabled:data-[state=checked]:bg-muted-foreground data-[state=unchecked]:border-muted-foreground data-[state=unchecked]:bg-transparent hover:not-disabled:data-[state=unchecked]:border-accent/80',
      className
    )}
    {...props}
    ref={ref}
  >
    <SwitchPrimitives.Thumb
      className={cn(
        'pointer-events-none block size-4 rounded-full bg-primary-foreground shadow-lg ring-0 transition-transform data-[state=checked]:translate-x-4 data-[state=unchecked]:translate-x-0.5 data-[state=unchecked]:bg-muted-foreground dark:data-[state=checked]:bg-background dark:data-[state=unchecked]:bg-foreground'
      )}
    />
  </SwitchPrimitives.Root>
));
Switch.displayName = SwitchPrimitives.Root.displayName;

export { Switch };
