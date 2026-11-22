/* Test shim: in an unprivileged user namespace setgroups() is denied and dnsmasq refuses to start because of it. */
#include <grp.h>
int setgroups(size_t n, const gid_t *list) { (void)n; (void)list; return 0; }
