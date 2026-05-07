# fullrmc Packmol Docker Link

SAXSShell can link a pre-existing Docker container that has Packmol installed.
In the `fullrmc` workflow, Packmol is the external packer that consumes the
generated `packmol_inputs` folder and writes the packed coordinate output for
downstream RMC preparation.

SAXSShell does **not** install Packmol for you and does **not** create the
Docker container automatically. The UI only validates the container, stores the
link metadata for the active project, and syncs generated Packmol input files
into a container-side project folder.

## What Packmol is

According to the official Packmol project, Packmol builds initial molecular
configurations by packing molecules into user-defined regions while enforcing
minimum-distance constraints between atoms from different molecules. The
upstream project documentation describes it as a tool for building initial
configurations for molecular dynamics simulations and notes support for `PDB`,
`TINKER`, and `XYZ` inputs.

For SAXSShell users, the practical role of Packmol is simpler:

- SAXSShell prepares the representative structures, composition plan, and
  `packmol_combined.inp` input file.
- Packmol is then run separately inside the linked container to build the
  packed coordinate output.
- Because the Packmol working folder is bind-mounted from the host, the input
  and output files remain visible to both the container and your local project
  environment.

## How to cite Packmol

The upstream Packmol README asks users to cite one of the Packmol papers when
the software contributes to a publication. For most SAXSShell workflows, the
general package paper is the clearest reference:

- L. Martinez, R. Andrade, E. G. Birgin, and J. M. Martinez,
  _Packmol: A package for building initial configurations for molecular
  dynamics simulations_,
  _Journal of Computational Chemistry_ **30** (2009), 2157-2164.
  DOI: <https://doi.org/10.1002/jcc.21224>

The older packing-optimization paper is also listed by Packmol upstream:

- J. M. Martinez and L. Martinez,
  _Packing optimization for the automated generation of complex system's
  initial configurations for molecular dynamics and docking_,
  _Journal of Computational Chemistry_ **24** (2003), 819-825.
  DOI: <https://doi.org/10.1002/jcc.10216>

If you describe the software tool itself, cite the 2009 paper first unless
your workflow specifically depends on the earlier algorithm paper.

## SAXSShell container convention

The Packmol Docker link in SAXSShell currently requires the selected
container-side project root to live under:

```text
/packmol_input_files
```

This is a SAXSShell convention enforced by the current UI. It is **not** a
general Docker requirement and **not** a Packmol requirement.

The recommended project layout inside the container is:

```text
/packmol_input_files/my_project
```

When a link is active, SAXSShell stores that project folder as the
container-side root and syncs generated Packmol inputs into:

```text
/packmol_input_files/my_project/rmcsetup/packmol_inputs
```

## Build a Docker image with Packmol installed

The example below follows the Packmol upstream manual compilation path
(`./configure` then `make`) inside a Debian-based image. Replace the release
tag if you want a different Packmol version.

```dockerfile
FROM debian:bookworm-slim

ARG PACKMOL_VERSION=21.2.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gfortran \
    git \
    make \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --branch "v${PACKMOL_VERSION}" --depth 1 \
    https://github.com/m3g/packmol.git /opt/packmol \
 && cd /opt/packmol \
 && ./configure \
 && make \
 && cp /opt/packmol/packmol /usr/local/bin/packmol

RUN mkdir -p /packmol_input_files
WORKDIR /packmol_input_files

# Keeps the container available for docker exec-based validation from SAXSShell.
CMD ["sleep", "infinity"]
```

Build the image:

```bash
docker build -t saxshell-packmol:21.2.1 -f Dockerfile.packmol .
```

!!! note "Alternative installation paths"
The Packmol upstream README also documents `pip install packmol` for many
platforms. The source-build path above is shown here because it follows the
upstream manual compilation instructions directly and keeps the container
behavior explicit.

## Bind-mount a host folder for read/write access

Docker's bind-mount documentation recommends the `--mount` flag and notes that
bind mounts are read-write by default. That default is the right choice for
this SAXSShell workflow, because:

- SAXSShell needs to copy Packmol input files into the container-visible folder
- Packmol needs to write packed output files back to the same host-backed
  folder

Create a host folder that will hold one or more Packmol-linked project roots:

```bash
mkdir -p "$HOME/saxshell_packmol_projects/my_project"
```

Start the container with that host folder mounted at
`/packmol_input_files`:

```bash
docker run -d \
  --name saxshell-packmol \
  --mount type=bind,src="$HOME/saxshell_packmol_projects",dst=/packmol_input_files \
  saxshell-packmol:21.2.1
```

Because the mount is read-write, files created inside
`/packmol_input_files/my_project` in the container will also appear under
`$HOME/saxshell_packmol_projects/my_project` on the host.

Verify that Packmol resolves and that the container-side project folder exists:

```bash
docker exec -it saxshell-packmol sh -lc \
  'packmol --help >/dev/null && mkdir -p /packmol_input_files/my_project && ls -ld /packmol_input_files/my_project'
```

!!! warning "Do not mount this path read-only"
A read-only bind mount is useful for inspection, but it will block the
normal SAXSShell Packmol workflow because the UI sync step and Packmol
itself both need write access to the mounted folder.

## Linking the container from the UI

Open either:

- the main SAXS window: `File > Link Packmol Docker Container...`
- the `fullrmc` window: `Tools > Link Packmol Docker Container`

The dialog lets you enter:

- a preset name for reuse across projects
- a discovered-container list pulled from Docker
- the Docker container name
- the Packmol command inside the container, usually `packmol`
- the shell command used for validation, usually `sh`
- the container-side project folder, which must be inside
  `/packmol_input_files`

After you press `Test Container`, SAXSShell will:

1. check that Docker is reachable
2. use the selected container name and start the container if it is not
   already running
3. verify that the selected folder exists inside the container
4. verify that the Packmol command resolves inside the container
5. capture a Packmol version/help line so the executable can be confirmed as
   runnable inside the container
6. load a directory tree in the dialog so you can pick the exact
   container-side project folder

!!! info "Image placeholder"
Add a screenshot of the **Link Packmol Docker Container** dialog after a
successful `Test Container`, with the container list, selected project
folder, and directory tree visible.

When the link is accepted, SAXSShell stores:

- a project-specific link file at `rmcsetup/packmol_docker_link.json`
- a reusable recent preset in the application settings so the same container
  can be linked to a new project more quickly later

## Running Packmol after SAXSShell syncs the inputs

When you run `Build Packmol Setup` in the `fullrmc` window:

1. SAXSShell writes the normal local `rmcsetup/packmol_inputs` files
2. if a Docker link is active, SAXSShell syncs those inputs into the linked
   container project folder
3. the Packmol section in the UI records the last sync status and the remote
   Packmol input/output paths

At that point, a typical Packmol run inside the container looks like:

```bash
docker exec -it saxshell-packmol sh -lc \
  'cd /packmol_input_files/my_project/rmcsetup/packmol_inputs && packmol < packmol_combined.inp'
```

Because the working folder is bind-mounted, the resulting packed output file is
available both inside the container and on the host filesystem.

## Manual startup reference

If the selected container is stopped, SAXSShell will try to start it before
validation. If your container only stays alive when attached to a terminal, the
manual reference command is:

```bash
docker start -i <container_name>
```

Use that command outside SAXSShell, then retry the link dialog.

## Practical notes

- Keep `/packmol_input_files` dedicated to the host bind mount so there is no
  confusion about where Packmol inputs and outputs are actually being written.
- Docker bind mounts obscure pre-existing files at the target path inside the
  container, so it is best to mount into a directory that is intentionally
  reserved for Packmol exchange files.
- If you are using a remote Docker daemon instead of a local one, bind mounts
  refer to paths on the Docker daemon host, not necessarily the machine where
  you launched the client command.

## References

- Packmol upstream project and installation instructions:
  <https://github.com/m3g/packmol>
- Packmol user guide and examples:
  <https://m3g.github.io/packmol>
- Docker bind mounts:
  <https://docs.docker.com/engine/storage/bind-mounts/>
- Docker `run` / `--mount` reference:
  <https://docs.docker.com/reference/cli/docker/container/run/>
