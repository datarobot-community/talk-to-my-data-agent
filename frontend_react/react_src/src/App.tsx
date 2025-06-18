import { useEffect } from "react";
import "./App.css";
import { Sidebar } from "@/components/Sidebar";
import Pages from "./pages";
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable";
import { useDataRobotInfo } from "./api-state/user/hooks";
import { useAppState } from "@/state/hooks";

function App() {
  useDataRobotInfo();
  const { theme } = useAppState();
  
  // Apply theme class on mount and theme change
  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);
  
  return (
    <div className="h-screen">
      <ResizablePanelGroup direction="horizontal">
        <ResizablePanel defaultSize={20} minSize={15} maxSize={30}>
          <Sidebar />
        </ResizablePanel>
        <ResizableHandle withHandle />
        <ResizablePanel defaultSize={80}>
          <Pages />
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
}

export default App;
