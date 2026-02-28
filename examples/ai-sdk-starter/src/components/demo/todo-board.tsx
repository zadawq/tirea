const buttonClass =
  "rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm font-semibold text-slate-800 transition hover:-translate-y-px hover:border-slate-400 hover:bg-slate-50";
const inputClass =
  "w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none ring-cyan-300 transition focus:ring-2";

export type Todo = {
  id: string;
  title: string;
  status: "pending" | "completed";
};

export function createTodo(index: number, title?: string): Todo {
  return {
    id: crypto.randomUUID(),
    title: title && title.trim().length > 0 ? title : `Task ${index + 1}`,
    status: "pending",
  };
}

export function TodoBoard({
  todos,
  onChange,
}: {
  todos: Todo[];
  onChange: (next: Todo[]) => void;
}) {
  const pending = todos.filter((t) => t.status === "pending");
  const completed = todos.filter((t) => t.status === "completed");

  const updateTodo = (id: string, patch: Partial<Todo>) => {
    onChange(todos.map((t) => (t.id === id ? { ...t, ...patch } : t)));
  };

  const deleteTodo = (id: string) => {
    onChange(todos.filter((t) => t.id !== id));
  };

  const addTodo = () => {
    onChange([...todos, createTodo(todos.length)]);
  };

  const renderColumn = (title: string, list: Todo[]) => (
    <div className="min-w-0 flex-1 rounded-2xl border border-slate-200 bg-white/90 p-3">
      <h3 className="text-base font-semibold text-slate-900">{title}</h3>
      <div className="mt-3 grid gap-2">
        {list.length === 0 && (
          <div className="rounded-lg border border-dashed border-slate-300 px-3 py-2 text-sm text-slate-500">
            No tasks
          </div>
        )}
        {list.map((todo) => (
          <div
            key={todo.id}
            data-testid="canvas-todo-card"
            className="rounded-xl border border-slate-300 bg-slate-50 p-2.5"
          >
            <div className="flex flex-wrap items-center gap-2">
              <button
                data-testid="canvas-toggle-status"
                className={buttonClass}
                onClick={() =>
                  updateTodo(todo.id, {
                    status: todo.status === "pending" ? "completed" : "pending",
                  })
                }
              >
                {todo.status === "pending" ? "Mark Done" : "Reopen"}
              </button>
              <button
                data-testid="canvas-delete-todo"
                className="rounded-lg border border-red-300 bg-red-50 px-3 py-1.5 text-sm font-semibold text-red-800 transition hover:-translate-y-px hover:border-red-400 hover:bg-red-100"
                onClick={() => deleteTodo(todo.id)}
              >
                Delete
              </button>
            </div>
            <input
              data-testid="canvas-title-input"
              value={todo.title}
              onChange={(e) => updateTodo(todo.id, { title: e.target.value })}
              className={`${inputClass} mt-2`}
            />
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="mt-3">
      <div className="flex flex-wrap items-center gap-2">
        <button data-testid="canvas-add-todo" className={buttonClass} onClick={addTodo}>
          Add Todo
        </button>
        <p data-testid="canvas-pending-count" className="m-0 text-sm text-slate-700">
          Pending: {pending.length}
        </p>
        <p data-testid="canvas-completed-count" className="m-0 text-sm text-slate-700">
          Completed: {completed.length}
        </p>
      </div>
      <div className="mt-3 flex flex-col gap-3 lg:flex-row">
        {renderColumn("To Do", pending)}
        {renderColumn("Done", completed)}
      </div>
    </div>
  );
}
