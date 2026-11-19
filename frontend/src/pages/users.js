

async function AddUser() {
  // BACKEND: PREFILLED LANG YUNG FORM SA BABA PARA MA MATCH SA DESIGN. Gawin nyong '' lahat ng initial values (at active: true) pag nilagay nyo na sa app.
  // BACKEND: Yung form field names ay fullName, email, username, role, password and active. Eto gamitin nyo pag mag se send ng create user request.
  const [values, setValues] = useState({
    fullName: 'Stephen Rey Bulahan',
    email: 'strbulahan@gmail.com',
    username: 'strbulahan',
    role: 'Traffic enforcer',
    password: 'TempPass#2026',
    active: true,
  });
  const [showPassword, setShowPassword] = useState(false);

  const handleChange = (event) => {
    const { name, type, value, checked } = event.target;
    setValues((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    // BACKEND: ise send nyo yung values sa create user endpoint dito.
    // BACKEND: Pag success i reset nyo yung form and i refresh yung USERS table. On failure show an error message (basic error handling lang yan).
    try {
      const response = await fetch("/api/v1/users/");
      if (!response.ok) {
        throw new Error(`Response status: ${response.status}`);
      }
      const result = await response.json();
      console.log(result);
    } catch (error) {
      console.error(error.message);
    }
  };

  return (
    <section className="panel">
      <h2 className="panel__title">Add user</h2>

      <form className="stack-form" onSubmit={handleSubmit}>
        <div className="field">
          <label className="field__label" htmlFor="fullName">Full name</label>
          <input
            className="field__input"
            id="fullName"
            name="fullName"
            type="text"
            value={values.fullName}
            onChange={handleChange}
            required
          />
        </div>

        <div className="field">
          <label className="field__label" htmlFor="email">Email</label>
          <input
            className="field__input"
            id="email"
            name="email"
            type="email"
            value={values.email}
            onChange={handleChange}
            required
          />
        </div>

        <div className="field">
          <label className="field__label" htmlFor="username">Username</label>
          <input
            className="field__input"
            id="username"
            name="username"
            type="text"
            value={values.username}
            onChange={handleChange}
            required
          />
        </div>

        <div className="field">
          <label className="field__label" htmlFor="role">Role</label>
          <select
            className="field__input field__select"
            id="role"
            name="role"
            value={values.role}
            onChange={handleChange}
          >
            {ROLES.map((role) => (
              <option key={role} value={role}>{role}</option>
            ))}
          </select>
        </div>

        <div className="field">
          <label className="field__label" htmlFor="password">Temporary password</label>
          <div className="field__control">
            <input
              className="field__input field__input--password"
              id="password"
              name="password"
              type={showPassword ? 'text' : 'password'}
              autoComplete="new-password"
              value={values.password}
              onChange={handleChange}
              required
            />
            <button
              type="button"
              className="icon-button"
              onClick={() => setShowPassword((prev) => !prev)}
              aria-label={showPassword ? 'Hide password' : 'Show password'}
            >
              <EyeIcon hidden={!showPassword} />
            </button>
          </div>
        </div>

        <div className="checkbox-list">
          <label className="checkbox">
            <input
              type="checkbox"
              name="active"
              checked={values.active}
              onChange={handleChange}
            />
            <span>Account active</span>
          </label>
        </div>

        {/* 
        BACKEND: Yung Save user ay nag su submit ng form. Yung Cancel naman dapat i clear yung form (lagyan nyo ng onClick).
      */}
        <div className="button-row">
          <button type="submit" className="action-button action-button--green action-button--full">
            Save user
          </button>
          <button type="button" className="action-button action-button--light action-button--full">
            Cancel
          </button>
        </div>
      </form>
    </section>
  );
}