'use client';

import * as React from 'react';
import * as TabsPrimitive from '@radix-ui/react-tabs';

import { cn } from '@/lib/utils';

function Tabs({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Root>) {
  return <TabsPrimitive.Root className={cn(className)} {...props} />;
}

function TabsList({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.List>) {
  return (
    <TabsPrimitive.List
      data-slot="tabs-list"
      className={cn(
        'inline-flex items-center rounded-lg border-2 border-sidebar-border bg-sidebar-border',
        className
      )}
      {...props}
    />
  );
}

function TabsTrigger({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Trigger>) {
  return (
    <TabsPrimitive.Trigger
      data-slot="tabs-trigger"
      className={cn(
        // Base styles
        'inline-flex items-center justify-center',
        // Spacing
        'px-4 py-1.5',
        // Typography
        'text-sm font-medium',
        // Background & Text
        'text-secondary-foreground',
        // Shape & Border
        'rounded-lg',
        // Cursor & Interactivity
        `
          cursor-pointer
          disabled:cursor-default
        `,
        // Focus state
        `
          outline-hidden
          focus:bg-muted focus:text-accent
          focus-visible:border-ring focus-visible:ring-[1px] focus-visible:ring-ring
        `,
        // Hover state
        'hover:bg-muted hover:text-accent',
        // Active/Pressed state
        `
          focus:z-2
          data-[state=active]:border-accent data-[state=active]:bg-sidebar-accent data-[state=active]:text-foreground
        `,
        // Disabled state
        'data-[disabled]:pointer-events-none data-[disabled]:text-muted-foreground data-[disabled]:opacity-50',
        // Text selection
        'select-none',
        // Transitions
        'transition-colors',
        className
      )}
      {...props}
    />
  );
}

function TabsContent({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Content>) {
  return (
    <TabsPrimitive.Content
      data-slot="tabs-content"
      className={cn('mt-2 outline-none', className)}
      {...props}
    />
  );
}

export { Tabs, TabsList, TabsTrigger, TabsContent };
