'use client';

import React, { useState, useRef, useEffect } from 'react';
import { cva } from 'class-variance-authority';
import { ChevronDown } from 'lucide-react';
import * as CollapsiblePrimitive from '@radix-ui/react-collapsible';
import { cn } from '@/lib/utils';

const COLLAPSIBLE_VARIANT = {
  section: 'section',
  standalone: 'standalone',
  panel: 'panel',
} as const;

const COLLAPSIBLE_VARIANTS = cva('group flex flex-col', {
  variants: {
    variant: {
      [COLLAPSIBLE_VARIANT.standalone]: 'rounded-lg shadow-sm',
      [COLLAPSIBLE_VARIANT.panel]: 'border-b',
      [COLLAPSIBLE_VARIANT.section]: '',
    },
  },
  defaultVariants: {
    variant: COLLAPSIBLE_VARIANT.panel,
  },
});

function Collapsible({
  className,
  variant = COLLAPSIBLE_VARIANT.panel,
  ...props
}: {
  variant?: (typeof COLLAPSIBLE_VARIANT)[keyof typeof COLLAPSIBLE_VARIANT];
} & React.ComponentProps<typeof CollapsiblePrimitive.Root>) {
  return (
    <CollapsiblePrimitive.Root
      data-slot="collapsible"
      data-variant={variant}
      className={cn(COLLAPSIBLE_VARIANTS({ variant, className }))}
      {...props}
    />
  );
}

function CollapsibleTrigger({
  className,
  ...props
}: React.ComponentProps<typeof CollapsiblePrimitive.CollapsibleTrigger>) {
  return (
    <CollapsiblePrimitive.CollapsibleTrigger
      data-slot="collapsible-trigger"
      className={cn(
        // layout
        'flex items-center justify-between',
        // spacing
        'p-3',
        // background
        'hover:bg-sidebar-primary',
        // animation
        'transition-all duration-300',
        // standalone variant styles
        'group-data-[variant=standalone]:bg-card',
        'group-data-[variant=standalone]:group-data-[state=closed]:rounded-lg',
        'group-data-[variant=standalone]:group-data-[state=open]:rounded-t',
        'group-data-[variant=standalone]:p-4',
        // section variant styles
        'group-data-[variant=section]:hover:bg-transparent',
        className
      )}
      {...props}
    />
  );
}

function CollapsibleContent({
  className,
  ...props
}: React.ComponentProps<typeof CollapsiblePrimitive.CollapsibleContent>) {
  return (
    <CollapsiblePrimitive.CollapsibleContent
      data-slot="collapsible-content"
      className={cn(
        'rounded-b',
        // standalone variant styles
        'group-data-[variant=standalone]:bg-card',
        className
      )}
      {...props}
    />
  );
}

function CollapsibleChevron({
  className,
  ...props
}: { className?: string } & React.SVGProps<SVGSVGElement>) {
  return (
    <ChevronDown
      className={cn(
        // animation
        `
          transition-transform duration-300
          group-data-[state=open]:rotate-180
        `,
        // size
        'size-4',
        className
      )}
      {...props}
    />
  );
}

function CollapsibleAnimatedContentWrapper({
  open,
  children,
  className,
  ...props
}: {
  open: boolean;
  className?: string;
  children: React.ReactNode;
} & React.HTMLAttributes<HTMLDivElement>) {
  const [height, setHeight] = useState('0');
  const [renderContent, setRenderContent] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  // When opening, render content first to measure height
  useEffect(() => {
    if (wrapperRef.current) {
      setHeight(`${wrapperRef.current.scrollHeight}px`);
    }

    if (open) {
      setRenderContent(true);
    } else if (!open && wrapperRef.current) {
      const rafId = requestAnimationFrame(() => setHeight('0'));

      const timer = setTimeout(() => setRenderContent(false), 300); // animation duration
      return () => {
        cancelAnimationFrame(rafId);
        clearTimeout(timer);
      };
    }
  }, [open]);

  useEffect(() => {
    if (wrapperRef.current && open) {
      setHeight(`${wrapperRef.current.scrollHeight}px`);
    }
  }, [renderContent]);

  return (
    <div
      className={cn('overflow-hidden transition-[max-height] duration-300 ease-in-out', className)}
      style={{ maxHeight: height }}
      {...props}
    >
      {renderContent && (
        <div
          ref={wrapperRef}
          className={`
            flex flex-col px-3 pb-3
            group-data-[variant=section]:p-0
            group-data-[variant=standalone]:px-4 group-data-[variant=standalone]:pb-4
          `}
        >
          {children}
        </div>
      )}
    </div>
  );
}

export {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
  CollapsibleChevron,
  CollapsibleAnimatedContentWrapper,
  COLLAPSIBLE_VARIANT,
};
