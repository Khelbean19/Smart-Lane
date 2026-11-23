(() => {
  const formatter = new Intl.DateTimeFormat('en-PH', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
    timeZone: 'Asia/Manila',
  });

  window.smartLaneClock = {
    format: (date) => `${formatter.format(date)} PHT`,
  };
})();
