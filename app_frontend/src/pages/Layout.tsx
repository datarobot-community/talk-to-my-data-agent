import { Sidebar } from '@/components/Sidebar';
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from '@/components/ui/resizable';
import { useState, useEffect } from 'react';
import { Outlet } from 'react-router-dom';

export const Layout = () => {
  const [sidebarWidth, setSidebarWidth] = useState(20);

  useEffect(() => {
    const handleResize = () => {
      const size = (window.innerWidth * sidebarWidth) / 100;
      document.documentElement.style.setProperty('--sidebar-width', `${size}px`);
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [sidebarWidth]);

  return (
    <div className="h-screen">
      <ResizablePanelGroup direction="horizontal">
        <ResizablePanel
          onResize={data => {
            setSidebarWidth(data);
            const size = (window.innerWidth * data) / 100;
            document.documentElement.style.setProperty('--sidebar-width', `${size}px`);
          }}
          defaultSize={20}
          minSize={20}
          maxSize={30}
        >
          <Sidebar />
        </ResizablePanel>
        <ResizableHandle withHandle />
        <ResizablePanel defaultSize={80}>
          <Outlet />
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
};
