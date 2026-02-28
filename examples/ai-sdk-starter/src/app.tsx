import { Routes, Route } from "react-router";
import { CanvasPage } from "./pages/canvas-page";
import { BasicPage } from "./pages/basic-page";
import { ThreadsPage } from "./pages/threads-page";

export function App() {
  return (
    <Routes>
      <Route path="/" element={<CanvasPage />} />
      <Route path="/basic" element={<BasicPage />} />
      <Route path="/threads" element={<ThreadsPage />} />
    </Routes>
  );
}
